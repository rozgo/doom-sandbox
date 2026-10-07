"""Build the Doom Sandbox journals and the home page into build/journals (ignored).

Each journal lives in site/journals/<id>/: journal.json (title, theme, hero, metrics, media list with
repository sources) and content.html (its chapters, as <section id=...><h2>...</h2>...</section>).
Media are referenced in the HTML as media/<key>.mp4, media/<key>.webp and media/<key>_poster.webp.
Videos are re-encoded for the web with the bundled imageio-ffmpeg (H.264, faststart, no audio) and
cached under build/journal_media; images become WebP. The home page (site/journals/hub/) links every
journal. Adapted from the MuJoCo Sandbox journals. Publishes nothing.

Usage: uv run --group site python scripts/build_journals.py [ID ...]
Publish: scripts/publish_pages.sh build/journals/<id> <id>; the home page with
         scripts/publish_pages.sh build/journals/hub .
"""

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

from doom_bert.overlay import font

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/"site/journals"
OUT = ROOT/"build/journals"
CACHE = ROOT/"build/journal_media"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TOKENS = ("bg", "text", "muted", "card", "line", "link", "chip", "soft", "bad", "bad-soft", "accent", "accent-2",
          "accent-3", "on-accent", "hero", "hero-text", "film", "metric", "toc", "tile-1", "tile-2", "tile-3", "tile-4",
          "tile-text")


def cached(key, suffix, make):
    """Run make(path) once per key; later builds reuse the file."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE/f"{hashlib.sha256(key.encode()).hexdigest()[:20]}{suffix}"
    if not path.exists():
        tmp = path.with_name(path.stem+".tmp"+suffix)
        make(tmp)
        tmp.rename(path)
    return path


def source(rel):
    path = ROOT/rel
    if not path.exists():
        raise FileNotFoundError(f"media source missing: {rel}")
    if path.stat().st_size < 1024 and path.read_bytes().startswith(b"version https://git-lfs"):
        raise RuntimeError(f"media source is a Git LFS pointer (run git lfs pull): {rel}")
    return path


def webp(src, out, max_width=1800):
    digest = hashlib.sha256(src.read_bytes()).hexdigest()

    def make(path):
        image = Image.open(src).convert("RGB")
        if image.width > max_width:
            image = image.resize((max_width, round(image.height*max_width/image.width)), Image.Resampling.LANCZOS)
        image.save(path, "WEBP", quality=84, method=6)
    shutil.copy2(cached(f"img:{digest}:{max_width}", ".webp", make), out)


def video(src, out, poster, spec):
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    width, crf = spec.get("max_width", 1600), spec.get("crf", 27)
    start, duration = spec.get("start"), spec.get("duration")
    trim = (["-ss", str(start)] if start is not None else [])+(["-t", str(duration)] if duration is not None else [])

    def encode(path):
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", *trim, "-i", str(src), "-an",
                        "-vf", f"scale='min({width},iw)':-2", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    encoded = cached(f"vid:{digest}:{width}:{crf}:{start}:{duration}", ".mp4", encode)
    shutil.copy2(encoded, out)
    if spec.get("poster"):
        webp(source(spec["poster"]), poster)
        return

    def frame(path):
        png = path.with_suffix(".png")
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", str(spec.get("poster_t", 1.0)), "-i", str(encoded),
                        "-frames:v", "1", str(png)], check=True)
        Image.open(png).convert("RGB").save(path, "WEBP", quality=84, method=6)
        png.unlink()
    shutil.copy2(cached(f"poster:{digest}:{width}:{crf}:{start}:{duration}:{spec.get('poster_t', 1.0)}", ".webp", frame),
                 poster)


def build_media(spec, folder):
    media = folder/"media"
    media.mkdir(parents=True, exist_ok=True)
    made = set()
    for key, item in spec.items():
        src = source(item["src"])
        if src.suffix.lower() in (".mp4", ".mov", ".gif"):
            video(src, media/f"{key}.mp4", media/f"{key}_poster.webp", item)
            made |= {f"media/{key}.mp4", f"media/{key}_poster.webp"}
        else:
            webp(src, media/f"{key}.webp", item.get("max_width", 1800))
            made.add(f"media/{key}.webp")
    return made


def tokens(theme):
    missing = [t for t in TOKENS if t not in theme["tokens"]]
    if missing:
        raise KeyError(f"theme tokens missing: {missing}")
    return " ".join(f"--{k}: {theme['tokens'][k]};" for k in TOKENS)


def hero_media(hero):
    key = hero["media"]
    if hero.get("kind", "video") == "image":
        return f'<img src="media/{key}.webp" alt="{html.escape(hero.get("alt", ""))}">'
    return (f'<video src="media/{key}.mp4" poster="media/{key}_poster.webp" controls muted playsinline '
            f'preload="metadata"></video>')


def toc(content):
    out = []
    for sid, title in re.findall(r'<section id="([^"]+)"[^>]*>\s*<h2>(.*?)</h2>', content, re.S):
        label = re.sub(r"<[^>]+>", "", title)
        out.append(f'<a href="#{sid}">{label.replace(" · ", " ")}</a>')
    return "".join(out)


def check_references(page, made, name):
    refs = set(re.findall(r'(?:src|poster|href)="(media/[^"]+)"', page))
    missing = sorted(refs-made)
    if missing:
        raise RuntimeError(f"{name}: media referenced but not built: {missing}")
    unused = sorted(m for m in made-refs if not m.endswith("_poster.webp"))
    if unused:
        print(f"  {name}: built but unused: {unused}")


SOCIAL = (1200, 630)  # Open Graph / Twitter large-card size


def site():
    return json.loads((SRC/"hub/hub.json").read_text())


def social_image(src, out, background):
    """Fit a poster into a 1200x630 JPEG (the format every link preview accepts), padding with the theme colour."""
    image = Image.open(src).convert("RGB")
    scale = min(SOCIAL[0]/image.width, SOCIAL[1]/image.height)
    image = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", SOCIAL, background)
    canvas.paste(image, ((SOCIAL[0]-image.width)//2, (SOCIAL[1]-image.height)//2))
    canvas.save(out, "JPEG", quality=86, optimize=True, progressive=True)


def social_meta(url, title, description, image, alt, kind="article"):
    """Open Graph and Twitter card tags; image URLs carry a content hash so previews refresh when it changes."""
    version = hashlib.sha256(image.read_bytes()).hexdigest()[:12]
    image_url = f"{url}{image.name}?v={version}"
    tags = [f'<link rel="canonical" href="{url}">',
            f'<meta property="og:type" content="{kind}">',
            f'<meta property="og:site_name" content="{html.escape(site()["site_name"])}">',
            f'<meta property="og:title" content="{title}">',
            f'<meta property="og:description" content="{description}">',
            f'<meta property="og:url" content="{url}">',
            f'<meta property="og:image" content="{image_url}">',
            '<meta property="og:image:type" content="image/jpeg">',
            f'<meta property="og:image:width" content="{SOCIAL[0]}">',
            f'<meta property="og:image:height" content="{SOCIAL[1]}">',
            f'<meta property="og:image:alt" content="{html.escape(alt)}">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{title}">',
            f'<meta name="twitter:description" content="{description}">',
            f'<meta name="twitter:image" content="{image_url}">',
            f'<meta name="twitter:image:alt" content="{html.escape(alt)}">']
    return "\n  ".join(tags)


def build_journal(jid):
    folder_src = SRC/jid
    spec = json.loads((folder_src/"journal.json").read_text())
    content = (folder_src/"content.html").read_text()
    out = OUT/jid
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    made = build_media(spec["media"], out)
    css = (SRC/"journal.css").read_bytes()
    (out/"journal.css").write_bytes(css)
    page = (SRC/"template.html").read_text()
    metrics = "".join(f'<div class="metric"><div class="value">{m["value"]}</div><div class="label">{m["label"]}</div></div>'
                      for m in spec["metrics"])
    url = f'{site()["site_url"]}{jid}/'
    social_image(source(spec["card"]["poster"]), out/"social.jpg", spec["theme"]["tokens"]["film"])
    social = social_meta(url, html.escape(spec["title"]), html.escape(spec["description"]), out/"social.jpg",
                         spec["heading"])
    fields = {"social": social, "title": html.escape(spec["title"]), "description": html.escape(spec["description"]),
              "theme_color": spec["theme"]["theme_color"], "scheme": spec["theme"]["scheme"],
              "css_version": hashlib.sha256(css).hexdigest()[:12], "tokens": tokens(spec["theme"]),
              "eyebrow": spec["eyebrow"], "heading": spec["heading"], "lede": spec["lede"],
              "hero_media": hero_media(spec["hero"]), "hero_caption": spec["hero"]["caption"], "metrics": metrics,
              "toc": toc(content), "scope": spec["scope"], "content": content, "footer": spec["footer"]}
    for k, v in fields.items():
        page = page.replace("{{"+k+"}}", v)
    leftover = re.findall(r"\{\{\w+\}\}", page)
    if leftover:
        raise RuntimeError(f"{jid}: unfilled template fields {leftover}")
    check_references(page, made, jid)
    (out/"index.html").write_text(page)
    (out/".nojekyll").write_text("")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{jid}: {len(made)} media files, {size/1e6:.1f} MB")
    return spec


def hub_social(spec, journals, out):
    """Home-page preview: the site title over the experiments' posters, each with its accent colour."""
    canvas = Image.new("RGB", SOCIAL, "#0f0b0a")
    draw = ImageDraw.Draw(canvas)
    for y in range(SOCIAL[1]):  # the hub hero's ember gradient, top to bottom
        t = y / SOCIAL[1]
        draw.line((0, y, SOCIAL[0], y), fill=tuple(round(a + (b - a) * t) for a, b in zip((15, 11, 10), (74, 26, 18), strict=True)))
    draw.text((60, 52), spec["site_name"], font=font(64, bold=True), fill="#f6efe9")
    draw.text((62, 134), spec["social_line"], font=font(28), fill="#e7d9cf")
    count = len(spec["projects"])
    gap, width = 28, (SOCIAL[0] - 120 - 28 * (count - 1)) // count
    height = round(width * 9 / 16)
    for i, entry in enumerate(spec["projects"]):
        j = journals.get(entry["id"]) or json.loads((SRC/entry["id"]/"journal.json").read_text())
        poster = Image.open(source(j["card"]["poster"])).convert("RGB")
        scale = max(width / poster.width, height / poster.height)
        poster = poster.resize((round(poster.width*scale), round(poster.height*scale)), Image.Resampling.LANCZOS)
        left, top = (poster.width - width)//2, (poster.height - height)//2
        x, y = 60 + i * (width + gap), 220
        canvas.paste(poster.crop((left, top, left + width, top + height)), (x, y))
        draw.rectangle((x, y + height, x + width, y + height + 6), fill=j["theme"]["tokens"]["accent"])
        draw.text((x, y + height + 22), entry["short"], font=font(26, bold=True), fill="#f6efe9")
        draw.text((x, y + height + 58), entry["detail"], font=font(20), fill="#d9c9be")
    canvas.save(out, "JPEG", quality=86, optimize=True, progressive=True)


def build_hub(journals):
    spec = json.loads((SRC/"hub/hub.json").read_text())
    out = OUT/"hub"
    if out.exists():
        shutil.rmtree(out)
    (out/"hub_media").mkdir(parents=True)
    cards = []
    for entry in spec["projects"]:
        j = journals.get(entry["id"]) or (json.loads((SRC/entry["id"]/"journal.json").read_text())
                                           if (SRC/entry["id"]/"journal.json").exists() else {})
        entry = {**j.get("card", {}), "href": f"{entry['id']}/", **entry}  # the hub entry overrides the journal's card
        poster = f"hub_media/{entry['id']}.webp"
        webp(source(entry["poster"]), out/poster, 1200)
        accent = entry.get("accent") or j["theme"]["tokens"]["accent"]
        band = entry.get("band") or j["theme"]["tokens"]["hero"]
        tags = "".join(f"<span>{html.escape(t)}</span>" for t in entry["tags"])
        cards.append(f'''    <a class="card" href="{entry["href"]}" style="--card-accent: {accent}; --card-band: {band};">
      <div class="poster"><img src="{poster}" alt="" loading="lazy"></div>
      <div class="body">
        <p class="kicker">{html.escape(entry["kicker"])}</p>
        <h2>{html.escape(entry["title"])}</h2>
        <p class="line">{entry["line"]}</p>
        <p class="tags">{tags}</p>
        <p class="go">Read the journal &rarr;</p>
      </div>
    </a>''')
    page = (SRC/"hub/index.html").read_text().replace("{{cards}}", "\n".join(cards))
    hub_social(spec, journals, out/"social.jpg")
    description = re.search(r'<meta name="description" content="([^"]*)"', page).group(1)
    page = page.replace("{{social}}", social_meta(spec["site_url"], html.escape(spec["site_name"]), description,
                                                  out/"social.jpg", spec["social_alt"], kind="website"))
    css = (SRC/"hub/hub.css").read_bytes()
    (out/"hub.css").write_bytes(css)
    page = page.replace("{{css_version}}", hashlib.sha256(css).hexdigest()[:12])
    (out/"index.html").write_text(page)
    (out/".nojekyll").write_text("")
    print(f"hub: {len(cards)} projects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    ids = args.ids or sorted(p.parent.name for p in SRC.glob("*/journal.json"))
    journals = {jid: build_journal(jid) for jid in ids}
    if not args.ids:
        build_hub(journals)


if __name__ == "__main__":
    main()
