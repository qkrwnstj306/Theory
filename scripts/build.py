#!/usr/bin/env python3
"""Generate the READMEs and the GitHub Pages site from catalog.yml.

    python scripts/build.py           # regenerate README files + check every path in catalog.yml
    python scripts/build.py site      # build the MkDocs site into _build/site
    python scripts/build.py serve     # live preview at http://127.0.0.1:8000
"""
import os
import posixpath
import re
import shutil
import subprocess
import sys
import urllib.parse

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BUILD = '_build'
STAGE = f'{BUILD}/docs'
CONTENT_DIRS = ['papers', 'notes', 'lectures', 'research', 'projects']
IMG_EXT = ('.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp')
GENERATED = '<!-- 이 파일은 scripts/build.py 가 catalog.yml 에서 자동 생성합니다. 직접 수정하지 말고 catalog.yml 을 고치세요. -->'

VENUE_RE = re.compile(r'^\*\s*(.+?)\s*\((\d{4})\)')
REVIEW_DATE_RE = re.compile(r'Review Data:\s*(\d{4})\.(\d{1,2})\.(\d{1,2})')
LINK_RE = re.compile(r"""((?:href|src)\s*=\s*)(['"])(.*?)\2|(\]\()([^)\s]+)((?:\s+"[^"]*")?\))""")

KIND = {  # icon, label
    'review': ('📝', '리뷰'),
    'notes': ('📒', '필기 PDF'),
    'slides': ('📊', '발표 자료'),
    'files': ('📎', '기타 자료'),
    'code': ('💻', '코드'),
    'links': ('🔗', '관련 링크'),
}


# --------------------------------------------------------------------------- catalog
def load_catalog():
    cat = yaml.safe_load(open('catalog.yml', encoding='utf-8'))
    fields = {f['id']: f for f in cat['fields']}
    for p in cat['papers']:
        if p['field'] not in fields:
            sys.exit(f"catalog.yml: unknown field '{p['field']}' in {p['title']}")
        _autofill(p)
    for r in cat['research']:
        _autofill(r)
    return cat


def _autofill(e):
    """review / venue / year / added are optional: derive them from the review markdown."""
    if e.get('review') is False:
        del e['review']
    elif 'review' not in e and os.path.exists(f"{e['dir']}/README.md"):
        e['review'] = f"{e['dir']}/README.md"
    if e.get('review') and os.path.exists(e['review']) and not all(k in e for k in ('venue', 'year', 'added')):
        head = open(e['review'], encoding='utf-8').read().split('\n')[:15]
        for line in head:
            m = VENUE_RE.match(line.strip())
            if m and m.group(1) not in ('?', '[paper]'):
                e.setdefault('venue', m.group(1))
                e.setdefault('year', int(m.group(2)))
            m = REVIEW_DATE_RE.search(line)
            if m:
                e.setdefault('added', f'{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}')
    e['added'] = str(e.get('added', ''))


def catalog_paths(cat):
    """Every local file path referenced by the catalog."""
    out = []
    for p in cat['papers'] + cat['research']:
        out += [p['review']] if p.get('review') else []
        out += p.get('notes', []) + p.get('slides', [])
        out += [f['path'] for f in p.get('files', [])]
    for g in cat['notes'] + cat['lectures']:
        for it in g['items']:
            out += [it['path']] if it.get('path') else []
            out += [f['path'] for f in it.get('files', []) if f.get('path')]
    for pr in cat['projects']:
        out += [f['path'] for f in pr.get('files', [])]
    return out


def check(cat):
    missing = [p for p in catalog_paths(cat) if not os.path.exists(p)]
    for p in missing:
        print(f'  ✗ missing: {p}')
    known = {p['dir'] for p in cat['papers']}
    for field in os.listdir('papers'):
        if not os.path.isdir(f'papers/{field}'):
            continue
        for d in os.listdir(f'papers/{field}'):
            if os.path.isdir(f'papers/{field}/{d}') and f'papers/{field}/{d}' not in known:
                print(f'  ! papers/{field}/{d} 폴더가 catalog.yml 에 없습니다')
    return not missing


def short(title):
    head = title.split(':')[0]
    return head if ':' in title and len(head) <= 32 else title


def sort_papers(ps):
    return sorted(ps, key=lambda p: (p.get('year') or 9999, p['title'].lower()))


def esc(s):
    return str(s).replace('|', '\\|')


# --------------------------------------------------------------------------- link rendering
class Linker:
    """Builds a link from the page being generated to a repo file.

    gh   : relative repo links (GitHub README rendering)
    site : .md -> relative .md (MkDocs resolves it), anything else -> GitHub blob URL
    """

    def __init__(self, cat, mode, page):
        self.mode, self.page = mode, page
        self.blob = cat['site']['repo'].rstrip('/') + '/blob/main/'

    def __call__(self, path):
        if self.mode == 'gh':
            return quote(posixpath.relpath(path, posixpath.dirname(self.page) or '.'))
        if path.endswith('.md'):
            return quote(posixpath.relpath(site_md(path), posixpath.dirname(self.page) or '.'))
        return self.blob + quote(path)


def quote(path):
    return urllib.parse.quote(path, safe="/.-_~()")


def site_md(path):
    return re.sub(r'(^|/)README\.md$', r'\1index.md', path)


def materials(p, L):
    out = []
    for n in p.get('notes', []):
        out.append(f'[{KIND["notes"][0]}]({L(n)} "{KIND["notes"][1]}")')
    for s in p.get('slides', []):
        out.append(f'[{KIND["slides"][0]}]({L(s)} "{KIND["slides"][1]}")')
    for f in p.get('files', []):
        out.append(f'[{KIND["files"][0]}]({L(f["path"])} "{f["label"]}")')
    for c in p.get('code', []):
        out.append(f'[{KIND["code"][0]}]({c} "{KIND["code"][1]}")')
    for l in p.get('links', []):
        out.append(f'[{KIND["links"][0]}]({l["url"]} "{l["label"]}")')
    return ' '.join(out)


def paper_table(papers, L, fields=None):
    head = '| Paper | Venue | Review | Materials |'
    sep = '|:--|:--:|:--:|:--|'
    if fields:
        head = '| Paper | Field | Venue | Review | Materials |'
        sep = '|:--|:--|:--:|:--:|:--|'
    rows = [head, sep]
    for p in papers:
        title = f'[{esc(p["title"])}]({p["url"]})' if p.get('url') else esc(p['title'])
        venue = f'{p.get("venue", "")} {p.get("year", "")}'.strip()
        review = ''
        if p.get('review'):
            review = f'[{KIND["review"][0]}]({L(p["review"])} "리뷰 읽기")' + (' ⚡' if p.get('simple') else '')
        cells = [title]
        if fields:
            f = fields[p['field']]
            cells.append(f'[{esc(f["name"])}]({L(f"papers/{f["id"]}/README.md")})')
        cells += [esc(venue), review, materials(p, L)]
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


LEGEND = '> 📝 리뷰 · ⚡ 짧은 리뷰 · 📒 필기 PDF · 📊 발표 자료 · 📎 기타 자료 · 💻 코드 · 🔗 관련 링크 — 제목을 누르면 원문으로 이동합니다.'


def item_list(group, L):
    lines = []
    for l in group.get('links', []):
        lines.append(f'> [{l["label"]}]({l["url"]})')
    if group.get('desc'):
        lines.append(f'> {group["desc"]}')
    if lines:
        lines.append('')
    for it in group['items']:
        href = L(it['path']) if it.get('path') else it['url']
        tag = '' if (it.get('path') or '').endswith('.md') or not it.get('path') else f' `{os.path.splitext(it["path"])[1][1:].upper()}`'
        extra = ''.join(f' · [{f["label"]}]({L(f["path"]) if f.get("path") else f["url"]})' for f in it.get('files', []))
        lines.append(f'- [{it["title"]}]({href}){tag}{extra}')
    return '\n'.join(lines)


def research_table(cat, L):
    rows = ['| Research | Venue | Materials |', '|:--|:--:|:--|']
    for r in cat['research']:
        title = f'[**{esc(r["title"])}**]({r["url"]})' if r.get('url') else f'**{esc(r["title"])}**'
        mats = []
        if r.get('review'):
            mats.append(f'[📝 소개]({L(r["review"])})')
        mats += [f'[{f["label"]}]({L(f["path"])})' for f in r.get('files', [])]
        rows.append(f'| {title} | {esc(r.get("venue", ""))} | {" · ".join(mats)} |')
    return '\n'.join(rows)


def stats(cat):
    papers = cat['papers']
    return {
        'papers': len(papers),
        'reviews': sum(1 for p in papers if p.get('review')),
        'fields': len(cat['fields']),
        'notes': sum(len(g['items']) for g in cat['notes']),
        'lectures': sum(len(g['items']) for g in cat['lectures']),
    }


def recent(cat, n=10):
    ps = [p for p in cat['papers'] if p.get('added')]
    return sorted(ps, key=lambda p: p['added'], reverse=True)[:n]


# --------------------------------------------------------------------------- GitHub READMEs
def badge(label, value, color):
    return f'![{label}](https://img.shields.io/badge/{quote(label)}-{quote(str(value))}-{color}?style=flat-square)'


def gh_readme(cat):
    s, site, L = stats(cat), cat['site'], Linker(cat, 'gh', 'README.md')
    fields = {f['id']: f for f in cat['fields']}
    by_field = {f['id']: [] for f in cat['fields']}
    for p in cat['papers']:
        by_field[p['field']].append(p)
    desc = quote(site['subtitle'])
    out = [GENERATED, '<div align="center">', '',
           f'<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4f46e5,100:06b6d4&height=200&section=header'
           f'&text={quote(site["title"])}&fontSize=72&fontColor=ffffff&fontAlignY=36&desc={desc}&descSize=18&descAlignY=58&animation=fadeIn" width="100%"/>',
           '',
           f'[![Website](https://img.shields.io/badge/Website-Theory-4f46e5?style=for-the-badge&logo=materialformkdocs&logoColor=white)]({site["site"]}) '
           f'[![Portfolio](https://img.shields.io/badge/Portfolio-qkrwnstj306-06b6d4?style=for-the-badge&logo=githubpages&logoColor=white)]({site["portfolio"]})',
           '',
           ' '.join([badge('papers', s['papers'], '6366f1'), badge('reviews', s['reviews'], '8b5cf6'),
                     badge('fields', s['fields'], '0ea5e9'), badge('notes', s['notes'] + s['lectures'], '14b8a6')]),
           '',
           '읽은 논문의 리뷰, 이론 정리, 강의 노트를 연구 분야별로 모아둔 공간입니다.<br>',
           f'수식 렌더링과 검색이 되는 **[웹사이트]({site["site"]})** 에서 보는 것을 추천합니다.',
           '', '</div>', '',
           '## 🔬 My Research', '', research_table(cat, L), '',
           '## 🆕 Recently Added', '', '| Date | Paper | Field |', '|:--|:--|:--|']
    for p in recent(cat, 8):
        link = L(p['review']) if p.get('review') else p.get('url', '')
        f = fields[p['field']]
        out.append(f'| {p["added"]} | [{esc(p["title"])}]({link}) | [{f["name"]}]({L(f"papers/{f["id"]}/README.md")}) |')
    out += ['', '## 🗺️ Research Fields', '', '| Field | Papers | 설명 |', '|:--|:--:|:--|']
    for f in cat['fields']:
        out.append(f'| [**{f["name"]}**]({L(f"papers/{f["id"]}/README.md")}) | {len(by_field[f["id"]])} | {f["desc"]} |')
    out += ['', '## 📚 Papers', '', LEGEND, '']
    for f in cat['fields']:
        ps = sort_papers(by_field[f['id']])
        out += ['<details>', f'<summary><b>{f["name"]}</b> · {len(ps)}편 — {f["desc"]}</summary>', '',
                paper_table(ps, L), '', '</details>', '']
    out += ['## 📒 Notes', '']
    for g in cat['notes']:
        out += ['<details>', f'<summary><b>{g["title"]}</b> · {len(g["items"])}</summary>', '', item_list(g, L), '', '</details>', '']
    out += ['## 🎓 Lectures', '']
    for g in cat['lectures']:
        out += ['<details>', f'<summary><b>{g["title"]}</b> · {len(g["items"])}</summary>', '', item_list(g, L), '', '</details>', '']
    out += ['## 🧪 Projects', '']
    for pr in cat['projects']:
        extra = ''.join(f' · [{f["label"]}]({L(f["path"])})' for f in pr.get('files', []))
        out.append(f'- [{pr["title"]}]({pr["url"]}){extra}')
    out += ['', '## 🔗 Helpful Links', '']
    for g in cat['links']:
        out.append(f'**{g["group"]}** — ' + ' · '.join(f'[{i["label"]}]({i["url"]})' for i in g['items']))
        out.append('')
    if cat.get('todo'):
        out += ['## ✅ To-do', ''] + [f'- [ ] {t}' for t in cat['todo']] + ['']
    out += how_to_add(cat)
    out += ['---', '',
            '<sub>이 README 는 <code>catalog.yml</code> 에서 <code>python scripts/build.py</code> 로 자동 생성됩니다. '
            '직접 수정하지 말고 <code>catalog.yml</code> 을 고치세요.</sub>', '']
    return '\n'.join(out)


def how_to_add(cat):
    field_ids = ' · '.join(f'`{f["id"]}`' for f in cat['fields'])
    return [
        '## ✍️ 새 논문 리뷰 추가하기', '',
        '<details>',
        '<summary><b>펼쳐서 보기</b> — 폴더 만들기 → catalog.yml 등록 → 빌드 → push</summary>', '',
        '예시: `diffusion` 분야에 **Rectified Flow** 리뷰를 추가하는 경우 (`<slug>` = `rectified-flow`)', '',
        '**1. 리뷰 파일 만들기** — 템플릿을 새 폴더의 `README.md` 로 복사', '',
        '```powershell',
        'mkdir papers/diffusion/rectified-flow',
        'copy templates/review.md papers/diffusion/rectified-flow/README.md',
        '```', '',
        '리뷰 맨 위 두 줄은 이 형식을 지켜주세요. 학회·연도·추가 날짜가 여기서 자동으로 추출됩니다.', '',
        '```markdown',
        '## Rectified Flow: Flow Straight and Fast',
        '*ICLR(2023), 900 citation, UT Austin, Review Data: 2026.11.20*',
        '```', '',
        '- 이미지: 같은 폴더에 두고 `./img1.png` 로 참조',
        '- 필기 PDF / 발표 자료: 같은 폴더에 `<slug>.pdf`, `<slug>-slides.pptx`', '',
        '**2. `catalog.yml` 의 `papers:` 에 항목 추가** (위치 상관없음, 표는 연도순 자동 정렬)', '',
        '```yaml',
        "- title: 'Rectified Flow: Flow Straight and Fast'",
        '  field: diffusion',
        '  url: https://arxiv.org/abs/2209.03003',
        '  dir: papers/diffusion/rectified-flow',
        '  # ↓ 있을 때만',
        '  notes: [papers/diffusion/rectified-flow/rectified-flow.pdf]',
        '  slides: [papers/diffusion/rectified-flow/rectified-flow-slides.pptx]',
        '  code: [https://github.com/qkrwnstj306/...]',
        '  simple: true   # 짧은 리뷰(⚡)',
        '```', '',
        f'`field` 에 쓸 수 있는 값: {field_ids}', '',
        '**3. README 재생성 (+ 사이트 미리보기)**', '',
        '```powershell',
        'python scripts/build.py          # README 재생성 + 누락 파일 검사',
        'python scripts/build.py serve    # (선택) http://127.0.0.1:8000 미리보기',
        '```', '',
        '**4. push** — 1~2분 뒤 웹사이트에 자동 반영', '',
        '```powershell',
        'git add -A',
        'git commit -m "Add Rectified Flow review"',
        'git push',
        '```', '',
        '</details>', '',
    ]


def gh_field_readme(cat, f, mode='gh'):
    page = f'papers/{f["id"]}/README.md' if mode == 'gh' else f'papers/{f["id"]}/index.md'
    L = Linker(cat, mode, page)
    ps = sort_papers([p for p in cat['papers'] if p['field'] == f['id']])
    head = [GENERATED] if mode == 'gh' else [f'<div class="field-head">{icon_svg(f["icon"])}<span>{len(ps)} papers</span></div>', '']
    back = [f'[← 전체 목록]({L("README.md")})', ''] if mode == 'gh' else []
    return '\n'.join(head + [f'# {f["name"]}', '', f'> {f["desc"]}', ''] + back + [LEGEND, '', paper_table(ps, L), ''])


def gh_papers_index(cat, mode='gh'):
    page = 'papers/README.md' if mode == 'gh' else 'papers/index.md'
    L = Linker(cat, mode, page)
    fields = {f['id']: f for f in cat['fields']}
    counts = {f['id']: 0 for f in cat['fields']}
    for p in cat['papers']:
        counts[p['field']] += 1
    out = [GENERATED] if mode == 'gh' else []
    out += ['# Papers', '']
    if mode == 'site':
        out += [field_cards(cat, counts, 'papers/index.md'), '']
    else:
        out += ['| Field | Papers | 설명 |', '|:--|:--:|:--|']
        out += [f'| [**{f["name"]}**]({L(f"papers/{f["id"]}/README.md")}) | {counts[f["id"]]} | {f["desc"]} |' for f in cat['fields']]
        out += ['']
    out += ['## 전체 논문', '', LEGEND, '', paper_table(sort_papers(cat['papers']), L, fields), '']
    return '\n'.join(out)


def listing_page(cat, key, title, mode='gh', intro=''):
    page = f'{key}/README.md' if mode == 'gh' else f'{key}/index.md'
    L = Linker(cat, mode, page)
    out = [GENERATED] if mode == 'gh' else []
    out += [f'# {title}', '']
    if intro:
        out += [intro, '']
    for g in cat[key]:
        out += [f'## {g["title"]}', '', item_list(g, L), '']
    return '\n'.join(out)


def research_page(cat, mode='gh'):
    page = 'research/README.md' if mode == 'gh' else 'research/index.md'
    L = Linker(cat, mode, page)
    out = [GENERATED] if mode == 'gh' else []
    out += ['# My Research', '']
    for r in cat['research']:
        title = f'[{r["title"]}]({r["url"]})' if r.get('url') else r['title']
        out += [f'## {title}', '']
        if r.get('venue'):
            out += [f'**{r["venue"]}**', '']
        if r.get('review'):
            out += [f'- [📝 소개 / 리뷰]({L(r["review"])})']
        out += [f'- [{f["label"]}]({L(f["path"])}) `{os.path.splitext(f["path"])[1][1:].upper()}`' for f in r.get('files', [])]
        out += ['']
    return '\n'.join(out)


def write(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    old = open(path, encoding='utf-8').read() if os.path.exists(path) else None
    if old != text:
        open(path, 'w', encoding='utf-8', newline='\n').write(text)
        print(f'  ✎ {path}')


def build_readmes(cat):
    write('README.md', gh_readme(cat))
    write('papers/README.md', gh_papers_index(cat))
    for f in cat['fields']:
        write(f'papers/{f["id"]}/README.md', gh_field_readme(cat, f))
    write('notes/README.md', listing_page(cat, 'notes', '📒 Notes'))
    write('lectures/README.md', listing_page(cat, 'lectures', '🎓 Lectures'))
    write('research/README.md', research_page(cat))


# --------------------------------------------------------------------------- site
def icon_svg(name):
    import material  # mkdocs-material ships the Material Design Icons as SVG files
    path = os.path.join(os.path.dirname(material.__file__), 'templates', '.icons', *name.split('/')) + '.svg'
    return f'<span class="twemoji">{open(path, encoding="utf-8").read()}</span>' if os.path.exists(path) else ''


def field_cards(cat, counts, page):
    cards = []
    for f in cat['fields']:
        href = posixpath.relpath(f'papers/{f["id"]}/', posixpath.dirname(page) or '.') + '/'
        cards.append(f'<a class="field-card" href="{href}">{icon_svg(f["icon"])}'
                     f'<span class="fc-name">{f["name"]}</span><span class="fc-count">{counts[f["id"]]}</span>'
                     f'<span class="fc-desc">{f["desc"]}</span></a>')
    return '<div class="field-grid">\n' + '\n'.join(cards) + '\n</div>'


def site_home(cat):
    s, site = stats(cat), cat['site']
    counts = {f['id']: 0 for f in cat['fields']}
    for p in cat['papers']:
        counts[p['field']] += 1
    fields = {f['id']: f for f in cat['fields']}
    L = Linker(cat, 'site', 'index.md')
    research = []
    for r in cat['research']:
        href = posixpath.dirname(site_md(r['review'])) + '/' if r.get('review') else 'research/'
        research.append(f'<a class="research-card" href="{href}"><span class="rc-venue">{r.get("venue", "")}</span>'
                        f'<span class="rc-title">{r["title"]}</span><span class="rc-meta">{len(r.get("files", []))} materials</span></a>')
    rec = ['| Date | Paper | Field |', '|:--|:--|:--|']
    for p in recent(cat, 10):
        link = L(p['review']) if p.get('review') else p.get('url', '')
        f = fields[p['field']]
        rec.append(f'| {p["added"]} | [{esc(p["title"])}]({link}) | [{f["name"]}]({L(f"papers/{f["id"]}/README.md")}) |')
    return '\n'.join([
        '---', 'hide:', '  - navigation', '  - toc', '---', '',
        '<div class="hero">',
        '<p class="hero-eyebrow">Paper Reviews · Theory Notes · Lectures</p>',
        f'<h1 class="hero-title">{site["title"]}</h1>',
        f'<p class="hero-sub">{site["subtitle"]}</p>',
        '<div class="hero-stats">',
        f'<div><b>{s["papers"]}</b><span>Papers</span></div>',
        f'<div><b>{s["reviews"]}</b><span>Reviews</span></div>',
        f'<div><b>{s["fields"]}</b><span>Fields</span></div>',
        f'<div><b>{s["notes"] + s["lectures"]}</b><span>Notes</span></div>',
        '</div>',
        '<div class="hero-actions">',
        '<a class="md-button md-button--primary" href="papers/">Papers 둘러보기</a>',
        f'<a class="md-button" href="{site["repo"]}">GitHub</a>',
        f'<a class="md-button" href="{site["portfolio"]}">Portfolio</a>',
        '</div>', '</div>', '',
        '## My Research', '', '<div class="research-grid">', *research, '</div>', '',
        '## Research Fields', '', field_cards(cat, counts, 'index.md'), '',
        '## Recently Added', '', *rec, '',
    ])


def site_links(cat):
    L = Linker(cat, 'site', 'links.md')
    out = ['# Links & Projects', '', '## Projects', '']
    for pr in cat['projects']:
        extra = ''.join(f' · [{f["label"]}]({L(f["path"])})' for f in pr.get('files', []))
        out.append(f'- [{pr["title"]}]({pr["url"]}){extra}')
    for g in cat['links']:
        out += ['', f'## {g["group"]}', ''] + [f'- [{i["label"]}]({i["url"]})' for i in g['items']]
    if cat.get('todo'):
        out += ['', '## To-do', ''] + [f'- [ ] {t}' for t in cat['todo']]
    return '\n'.join(out) + '\n'


def paper_meta(cat, p, page):
    """Chip bar placed on top of each review page in the site."""
    L = Linker(cat, 'site', page)
    f = {x['id']: x for x in cat['fields']}[p['field']]
    chips = []
    venue = f'{p.get("venue", "")} {p.get("year", "")}'.strip()
    if venue:
        chips.append(f'<span class="pm-venue">{venue}</span>')
    field_href = posixpath.relpath(f'papers/{f["id"]}', posixpath.dirname(page)) + '/'
    chips.append(f'<a class="pm-chip pm-field" href="{field_href}">{f["name"]}</a>')
    if p.get('url'):
        chips.append(f'<a class="pm-chip" href="{p["url"]}">📄 Paper</a>')
    for n in p.get('notes', []):
        chips.append(f'<a class="pm-chip" href="{L(n)}">📒 필기 PDF</a>')
    for s in p.get('slides', []):
        chips.append(f'<a class="pm-chip" href="{L(s)}">📊 발표 자료</a>')
    for x in p.get('files', []):
        chips.append(f'<a class="pm-chip" href="{L(x["path"])}">📎 {x["label"]}</a>')
    for c in p.get('code', []):
        chips.append(f'<a class="pm-chip" href="{c}">💻 Code</a>')
    for l in p.get('links', []):
        chips.append(f'<a class="pm-chip" href="{l["url"]}">🔗 {l["label"]}</a>')
    return '<div class="paper-meta">' + ''.join(chips) + '</div>\n\n'


def transform_md(cat, src, text):
    """Rewrite links of a staged markdown page so they work on the site."""
    page = site_md(src)
    page_dir = posixpath.dirname(page)
    blob = cat['site']['repo'].rstrip('/') + '/blob/main/'

    def fix(target, html):
        if not target or re.match(r'^(https?:|mailto:|#|data:|javascript:)', target):
            return target
        path, _, frag = target.partition('#')
        raw = urllib.parse.unquote(path)
        repo_path = posixpath.normpath(raw.lstrip('/') if raw.startswith('/') else posixpath.join(posixpath.dirname(src), raw))
        frag = '#' + frag if frag else ''
        if repo_path.lower().endswith(IMG_EXT):
            return target
        if repo_path.endswith('.md'):
            t = site_md(repo_path)
            if html:   # raw HTML is not rewritten by MkDocs -> point to the directory URL
                t_url = posixpath.dirname(t) if t.endswith('index.md') else t[:-3]
                return quote(posixpath.relpath(t_url, page_dir) + '/') + frag
            return quote(posixpath.relpath(t, page_dir)) + frag
        if os.path.isdir(repo_path) or not posixpath.splitext(repo_path)[1]:
            return target
        return blob + quote(repo_path) + frag

    def sub(m):
        if m.group(1):
            return m.group(1) + m.group(2) + fix(m.group(3), True) + m.group(2)
        return m.group(4) + fix(m.group(5), False) + m.group(6)

    return github_alerts(separate_display_math(normalize_lists(LINK_RE.sub(sub, text))))


LIST_RE = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')


def normalize_lists(text):
    """GitHub accepts 2- or 4-space nested lists and lists glued to a paragraph;
    Python-Markdown needs 4-space nesting and a blank line before a list."""
    out, stack, fenced = [], [], False      # stack: original indents of open list items
    for line in text.split('\n'):
        s = line.strip()
        if s.startswith('```'):
            fenced = not fenced
        if fenced or s.startswith('```'):
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        m = LIST_RE.match(line)
        if m:
            while stack and stack[-1] >= indent:
                if stack[-1] == indent:
                    stack.pop()
                    break
                stack.pop()
            if not stack and out and out[-1].strip() and not LIST_RE.match(out[-1]) \
                    and not out[-1].startswith(' '):
                out.append('')               # list glued to a paragraph
            out.append(' ' * (4 * len(stack)) + line.lstrip())
            stack.append(indent)
        elif not s:
            out.append(line)
        elif stack and indent > 0:           # continuation of the current item
            out.append(' ' * (4 * len(stack)) + line.lstrip())
        else:
            if indent == 0 and out and not out[-1].strip():
                stack = []                   # paragraph after a blank line ends the list
            out.append(line)
    return '\n'.join(out)


ALERTS = {'NOTE': ('note', 'Note'), 'TIP': ('tip', 'Tip'), 'IMPORTANT': ('abstract', 'Important'),
          'WARNING': ('warning', 'Warning'), 'CAUTION': ('danger', 'Caution')}


def github_alerts(text):
    """GitHub `> [!NOTE]` alert blocks -> Material admonitions."""
    out, lines, i = [], text.split('\n'), 0
    while i < len(lines):
        m = re.match(r'^(\s*)>\s*\[!(\w+)\]\s*$', lines[i])
        if m and m.group(2).upper() in ALERTS:
            indent, (kind, title) = m.group(1), ALERTS[m.group(2).upper()]
            if out and out[-1].strip():
                out.append('')
            out.append(f'{indent}!!! {kind} "{title}"')
            out.append('')
            i += 1
            while i < len(lines) and re.match(rf'^{indent}>', lines[i]):
                body = re.sub(rf'^{indent}>\s?', '', lines[i])
                out.append(f'{indent}    {body}' if body.strip() else '')
                i += 1
            out.append('')
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


def insert_after_title(text, block):
    """Put the chip bar under `# Title` + its `*venue*` line when the review has an H1."""
    lines = text.split('\n')
    for i, line in enumerate(lines[:10]):
        if line.startswith('# '):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and re.match(r'^\*.*\*\s*$', lines[j].strip()):
                i = j
            return '\n'.join(lines[:i + 1] + ['', block.rstrip('\n'), ''] + lines[i + 1:])
    return block + text


def separate_display_math(text):
    """GitHub renders `$$...$$` even when glued to a list line; Python-Markdown needs it
    to be its own paragraph, so pad display-math lines with blank lines (outside code)."""
    out, fenced = [], False
    lines = text.split('\n')
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('```'):
            fenced = not fenced
        if not fenced and s.startswith('$$') and out and out[-1].strip():
            # indent to the content column of the list item / paragraph it belongs to
            prev = out[-1]
            m = LIST_RE.match(prev)
            indent = len(m.group(1)) + 4 if m else len(prev) - len(prev.lstrip())
            line = ' ' * indent + s
            out.append('')
        out.append(line)
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
        if not fenced and s.endswith('$$') and len(s) > 2 and nxt:
            out.append('')
    return '\n'.join(out)


def stage(cat):
    shutil.rmtree(STAGE, ignore_errors=True)
    reviews = {p['review']: p for p in cat['papers'] if p.get('review')}
    for d in CONTENT_DIRS:
        for root, _, files in os.walk(d):
            for name in files:
                src = posixpath.join(root.replace(os.sep, '/'), name)
                if name.lower().endswith(IMG_EXT):
                    dst = f'{STAGE}/{src}'
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copyfile(src, dst)
                elif name.endswith('.md'):
                    text = transform_md(cat, src, open(src, encoding='utf-8').read())
                    if src in reviews:
                        text = insert_after_title(text, paper_meta(cat, reviews[src], site_md(src)))
                    dst = f'{STAGE}/{site_md(src)}'
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    open(dst, 'w', encoding='utf-8').write(text)
    shutil.copytree('website', STAGE, dirs_exist_ok=True)

    pages = {
        'index.md': site_home(cat),
        'papers/index.md': gh_papers_index(cat, 'site'),
        'notes/index.md': listing_page(cat, 'notes', 'Notes', 'site'),
        'lectures/index.md': listing_page(cat, 'lectures', 'Lectures', 'site'),
        'research/index.md': research_page(cat, 'site'),
        'links.md': site_links(cat),
    }
    for f in cat['fields']:
        pages[f'papers/{f["id"]}/index.md'] = gh_field_readme(cat, f, 'site')
    for path, text in pages.items():
        os.makedirs(os.path.dirname(f'{STAGE}/{path}'), exist_ok=True)
        open(f'{STAGE}/{path}', 'w', encoding='utf-8').write(text)

    nav = [{'Home': 'index.md'}]
    research = ['research/index.md'] + [{short(r['title']): site_md(r['review'])} for r in cat['research'] if r.get('review')]
    nav.append({'My Research': research})
    papers = ['papers/index.md']
    for f in cat['fields']:
        ps = sort_papers([p for p in cat['papers'] if p['field'] == f['id'] and p.get('review')])
        papers.append({f['name']: [f'papers/{f["id"]}/index.md'] + [{short(p['title']): site_md(p['review'])} for p in ps]})
    nav.append({'Papers': papers})
    notes = ['notes/index.md']
    for g in cat['notes']:
        mds = [{it['title']: site_md(it['path'])} for it in g['items'] if (it.get('path') or '').endswith('.md')]
        if mds:
            notes.append({g['title']: mds})
    nav.append({'Notes': notes})
    lectures = ['lectures/index.md'] + [{g['title']: site_md(it['path'])} for g in cat['lectures']
                                         for it in g['items'] if (it.get('path') or '').endswith('.md')]
    nav.append({'Lectures': lectures})
    nav.append({'Links': 'links.md'})

    cfg = {'INHERIT': '../mkdocs.yml', 'docs_dir': 'docs', 'site_dir': 'site', 'nav': nav}
    with open(f'{BUILD}/mkdocs.yml', 'w', encoding='utf-8') as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)


# --------------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'readme'
    cat = load_catalog()
    if cmd == 'readme':
        ok = check(cat)
        build_readmes(cat)
        print('README 생성 완료' + ('' if ok else ' (경고: 누락된 파일이 있습니다)'))
        sys.exit(0 if ok else 1)
    stage(cat)
    if cmd == 'site':
        subprocess.run([sys.executable, '-m', 'mkdocs', 'build', '-f', f'{BUILD}/mkdocs.yml'], check=True)
    elif cmd == 'serve':
        subprocess.run([sys.executable, '-m', 'mkdocs', 'serve', '-f', f'{BUILD}/mkdocs.yml'], check=True)
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
