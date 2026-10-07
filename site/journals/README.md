# Journals

Each Doom Sandbox experiment has a journal: its story, its videos and its measured results. The home
page (`hub/`) links them all. The format and build are adapted from the MuJoCo Sandbox journals.

```sh
uv run python scripts/journal_figures.py                    # figures drawn from committed reports
uv run --group site python scripts/build_journals.py        # every journal and the home page
uv run --group site python scripts/build_journals.py gliner2  # one journal
scripts/publish_pages.sh build/journals/gliner2 gliner2     # publish one journal folder
scripts/publish_pages.sh build/journals/hub .               # publish the home page at the site root
```

Built pages go to `build/journals/` (ignored). For a local preview with working home-page links, copy
`build/journals/hub/` and the journal folders into one directory, the way they are published, and serve
it with `python3 -m http.server`.

## A journal

`site/journals/<id>/journal.json`:

| Field | Content |
| --- | --- |
| `title` | Browser title, two to four words |
| `heading` | The page's h1: what the experiment is, in one line |
| `eyebrow` | `Engineering journal · <dates> · <main tools>` |
| `lede` | Two or three sentences: the model, the task, what is trained, zero-shot or scripted |
| `description` | One sentence for search results |
| `hero` | `{"media": key, "caption": text}`; add `"kind": "image"` and `"alt"` for an image |
| `metrics` | Four or five `{"value", "label"}` tiles: the measured headline numbers |
| `scope` | One short paragraph of HTML: where the experiment stands and what it does not show |
| `footer` | HTML: source folders, model licences and attribution |
| `card` | For the home page: `{"kicker", "line", "tags": [...], "poster": 16:9 repository image}` |
| `media` | `{key: {"src": repository path, optional "poster", "poster_t" (s), "start", "duration", "max_width", "crf"}}` |
| `theme` | `{"scheme": "dark" or "light", "theme_color", "tokens": {...}}` |

`site/journals/<id>/content.html` holds the chapters only, each
`<section id="slug"><h2>N · Title</h2><p class="when">date · one-line context</p> ... </section>`.
The table of contents is read from these. Components (`prose`, `film`, `grid2`, `lesson`, `failure`,
`note-box`, `result-strip`, `spec-cards`, `table-wrap`, `next-steps`) are styled by `journal.css`; see
the existing journals for examples.

## Rules

- Every number comes from a file in the repository (reports, summaries, model cards, READMEs); write
  what was measured, with its conditions. Keep failures on record. Say which parts are trained,
  zero-shot or fixed rules, and state each video's playback speed.
- Plain, concise, public-safe: no machine names or connection details, credentials, or links to
  private material.
- Videos are re-encoded for the web; keep a journal under about 70 MB built.

## Palettes

| Journal | Scheme | Character | Key colours |
| --- | --- | --- | --- |
| ModernBERT policy | dark | charcoal dashboard, signal cyan, amber | `#111316` `#1b1e23` `#edf0f5` `#73caff` `#f3bd72` |
| GLiNER2.5 comparison | light | lab paper, deep teal, brick | `#f4f1ea` `#1c1a16` `#0b8f7a` `#c0492f` |
| EmbeddingGemma 2 | dark | night violet, lilac, mint, amber | `#100e17` `#1a1724` `#a596ff` `#7fdc9a` `#f3bd72` |

The home page uses an ember-dark hero over paper; each card carries its journal's accent.
