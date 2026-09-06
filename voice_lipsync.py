#!/usr/bin/env python3
"""
voice_lipsync.py - the AUDIO layer for FilmForge. Turns SILENT rendered clips
into a film with SOUND: per-shot dialogue (TTS / voice-clone), lip-sync onto the
rendered face, optional background music, then mux + (optionally) stitch.

PIPELINE POSITION
  render_film.py   ->   voice_lipsync.py   ->   final film (with audio)
   (silent mp4s)         (voice + lipsync + music)

WHY: render_film/ComfyUI graphs output mute video. A film needs dialogue, voice,
and lip motion. This bolts that on AFTER rendering, so nothing about the render
path changes. filmforge.stitch keeps audio only when EVERY clip has an audio
stream, so before stitching we normalize silent (no-dialogue) shots to carry a
silent track - otherwise one mute shot drops sound from the whole film.

PLUGGABLE BACKENDS (all local / free - use whatever is installed):
  FF_TTS_CMD      shell template -> synthesize speech to a wav.
                  placeholders: {text} {ref} {out}
                  e.g. Piper : 'echo {text} | piper -m en.onnx -f {out}'
                  e.g. XTTS  : 'tts --text {text} --speaker_wav {ref} --out_path {out}'
  FF_LIPSYNC_CMD  shell template -> lip-sync a video to a wav.
                  placeholders: {video} {audio} {out}
                  e.g. LatentSync / Wav2Lip CLI.
GRACEFUL DEGRADATION
  - no FF_TTS_CMD    : with --dry-run, writes a placeholder tone track of the
                       right length so the rest of the chain is testable.
  - no FF_LIPSYNC_CMD: audio is MUXED onto the clip without mouth motion (still
                       a valid film with sound; add lip motion later).

SHOTS FILE (audio fields; render_film ignores them, this script consumes them)
  {
    "voices": { "hero": "voices/hero_ref.wav" },   # ref voice per character
    "music": "music/theme.mp3",                     # optional bg track (or null)
    "music_volume": 0.18,
    "shots": [
      {"id":"02", "dialogue":"Who's there?", "voice":"hero", ...}
    ]
  }

USAGE
  python3 voice_lipsync.py --shots shots.json
  python3 voice_lipsync.py --shots shots.json --stitch --out film.mp4 --transition crossfade
  python3 voice_lipsync.py --shots shots.json --dry-run --stitch --out film.mp4   # offline smoke
"""
import argparse, glob, json, os, pathlib, shlex, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import filmforge as ff                 # reuse stitch + _duration

DRY = False


def sh(cmd):
    """Run a command; list => no shell, str => shell. Raises on failure."""
    shell = isinstance(cmd, str)
    subprocess.run(cmd, shell=shell, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def ffprobe_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:np=1", str(path)],
            capture_output=True, text=True)
        return float((out.stdout or "").strip() or 0.0)
    except Exception:
        return 0.0


def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return "audio" in (out.stdout or "")


def placeholder_wav(seconds, out, tone=True):
    seconds = max(0.2, float(seconds))
    src = (f"sine=frequency=220:duration={seconds}" if tone
           else "anullsrc=r=24000:cl=mono")
    sh(["ffmpeg", "-y", "-f", "lavfi", "-i", src, "-t", str(seconds),
        "-ac", "1", "-ar", "24000", str(out)])


def synth(text, ref, out, seconds):
    tpl = os.environ.get("FF_TTS_CMD")
    if tpl:
        cmd = (tpl.replace("{text}", shlex.quote(text))
                  .replace("{ref}", shlex.quote(ref or ""))
                  .replace("{out}", shlex.quote(str(out))))
        sh(cmd)
        return "tts"
    # no TTS backend
    placeholder_wav(seconds, out, tone=DRY)
    return "placeholder" if DRY else "placeholder(no FF_TTS_CMD)"


def mux(video, audio, out):
    sh(["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(out)])


def lipsync(video, audio, out):
    tpl = os.environ.get("FF_LIPSYNC_CMD")
    if tpl:
        cmd = (tpl.replace("{video}", shlex.quote(str(video)))
                  .replace("{audio}", shlex.quote(str(audio)))
                  .replace("{out}", shlex.quote(str(out))))
        sh(cmd)
        return "lipsync"
    mux(video, audio, out)
    return "mux(no FF_LIPSYNC_CMD)"


def add_music(video, music, out, vol):
    if has_audio(video):
        fc = (f"[1:a]volume={vol}[m];"
              f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[aout]")
        sh(["ffmpeg", "-y", "-i", str(video), "-i", str(music),
            "-filter_complex", fc, "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", str(out)])
    else:
        sh(["ffmpeg", "-y", "-i", str(video), "-i", str(music),
            "-map", "0:v:0", "-map", "1:a:0", "-af", f"volume={vol}",
            "-shortest", "-c:v", "copy", "-c:a", "aac", str(out)])


def ensure_audio(clip, clips_dir):
    """Guarantee an audio stream so MIXED silent/voiced clips stitch WITH sound.
    filmforge.stitch keeps audio only if EVERY clip has it; a single mute shot
    would otherwise drop sound from the entire film."""
    if has_audio(clip):
        return clip
    dur = ffprobe_duration(clip) or 1.0
    out = pathlib.Path(clips_dir) / (pathlib.Path(clip).stem + "_silaud.mp4")
    sh(["ffmpeg", "-y", "-i", str(clip), "-f", "lavfi", "-i",
        "anullsrc=r=24000:cl=stereo", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-t", str(dur), str(out)])
    return str(out)


def find_clip(clips_dir, shot_id):
    hits = sorted(glob.glob(str(pathlib.Path(clips_dir) /
                                f"shot_{shot_id}_*.mp4")))
    hits = [h for h in hits
            if not h.endswith(("_voiced.mp4", "_final.mp4", "_silaud.mp4"))]
    return hits[0] if hits else None


def process(shots_path, clips_dir, music_default=None, music_vol=0.18):
    spec = json.loads(pathlib.Path(shots_path).read_text())
    voices = spec.get("voices", {}) or {}
    music = spec.get("music", music_default)
    music_vol = float(spec.get("music_volume", music_vol))
    clips_dir = pathlib.Path(clips_dir)
    results = []
    for shot in spec["shots"]:
        sid = shot.get("id", "x")
        clip = find_clip(clips_dir, sid)
        if not clip:
            print(f"  [warn] shot {sid}: no rendered clip in {clips_dir} - skip")
            continue
        work = clip
        dialogue = shot.get("dialogue")
        if dialogue:
            secs = ffprobe_duration(clip) or float(shot.get("seconds", 3))
            ref = voices.get(shot.get("voice"))
            wav = clips_dir / f"shot_{sid}_voice.wav"
            how_tts = synth(dialogue, ref, wav, secs)
            voiced = clips_dir / f"shot_{sid}_voiced.mp4"
            how_ls = lipsync(clip, wav, voiced)
            work = voiced
            print(f"  [shot {sid}] dialogue -> {how_tts} + {how_ls}")
        else:
            print(f"  [shot {sid}] no dialogue (kept silent)")
        if music:
            final = clips_dir / f"shot_{sid}_final.mp4"
            add_music(work, music, final, music_vol)
            work = final
        results.append(str(work))
    return results


def main():
    global DRY
    ap = argparse.ArgumentParser(description="Add voice+lipsync(+music) to rendered clips")
    ap.add_argument("--shots", required=True)
    ap.add_argument("--clips-dir",
                    default=str(pathlib.Path(os.path.expanduser("~")) /
                               ".filmforge" / "clips"))
    ap.add_argument("--stitch", action="store_true",
                    help="stitch the voiced clips into one film")
    ap.add_argument("--out", default="film_audio.mp4")
    ap.add_argument("--transition", choices=["cut", "crossfade"], default="cut")
    ap.add_argument("--dry-run", action="store_true",
                    help="no TTS backend needed; write placeholder audio (smoke test)")
    args = ap.parse_args()
    DRY = args.dry_run

    print(f"voice_lipsync: shots={args.shots} clips={args.clips_dir} "
          f"tts={'FF_TTS_CMD' if os.environ.get('FF_TTS_CMD') else ('dry' if DRY else 'none')} "
          f"lipsync={'FF_LIPSYNC_CMD' if os.environ.get('FF_LIPSYNC_CMD') else 'mux'}\n")
    clips = process(args.shots, args.clips_dir)
    if not clips:
        sys.exit("voice_lipsync: no clips produced")
    if args.stitch:
        # normalize: every clip must carry audio or stitch drops sound entirely
        clips = [ensure_audio(c, args.clips_dir) for c in clips]
        print(f"\nStitching {len(clips)} voiced clips -> {args.out} ({args.transition})")
        ff.stitch(clips, args.out, transition=args.transition)
        print(f"Done. Final film w/ audio: {args.out} "
              f"({ff._duration(args.out):.2f}s, audio={has_audio(args.out)})")
    else:
        for c in clips:
            print(f"  voiced clip: {c}  (audio={has_audio(c)})")


if __name__ == "__main__":
    main()
