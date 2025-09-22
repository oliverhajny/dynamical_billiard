# dynamical_billiard

Simulace dynamického biliardu (elipsa, Bunimovičův stadion) s podporou Poincarého map a animací. Projekt nabízí CLI i jednoduché API pro spouštění trajektorií a ukládání výstupů.

## Zadání problému
Studujeme pohyb hmotného bodu (kuličky) uvnitř ohraničeného tvaru (elipsa nebo Bunimovičův stadion). Kulička má konstantní energii, při nárazu do stěny žádnou neztrácí a odráží se zrcadlově (úhel odrazu se rovná úhlu dopadu). Mezi nárazy se pohybuje přímočaře konstantní rychlostí. Cílem je:
- simulovat trajektorie v dvou základních tvarech (elipsa a stadion),
- zkoumat chaotické chování systému
- generovat Poincarého mapy systému,
- vizualizovat průběh (animace trajektorie) a ukládat výsledky.

Poincarého mapa je způsob, jak zaznamenat do 2D obrázku průběh složitějšího (více dimenzionálního) dynamického systému. V našem případě Poincarého mapa ukazuje pouze ty stavy, kdy se kulička dotkne hrany. Využíváme Birkhoffových souřadnic (s, p), kde 's' představuje pozici na obvodu billiardu a 'p' sinus úhlu mezi směrovým vektorem a tečnou v bodě dopadu.

Bunimovičův stadion je tvořen dvěma stejně dlouhými rovnoběžnými úsečkami spojenými polokružnicemi. Na rozdíl od eliptického biliardu stadion vykazuje chaotické chování (jak lze vidět z Poincarého map).

## Použití

Požadavky: Python 3.10+, balíčky `numpy`, `matplotlib` (pro testy `pytest`).

Instalace (doporučeno virtuální prostředí):
- macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
- Windows (PowerShell): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
- Balíčky: `pip install numpy matplotlib pytest`

Spuštění z příkazové řádky (CLI) (jako modul):
Default run spustí vícero simulucí naráz s různými počátečními pozicemi a počátečním směrem a vygeneruje Poincarého mapu všech těchto trajektorií. Níže jsou vypsány všechny argumenty, které lze do CLI zadat a upravit tak tvar billiardu nebo jiné vlastnosti. K vygenerování animace trajektorie (v případě vícero simulací se vygeneruje první trajektorie) slouží argument --animate.

- Poznámka: při „src“ layoutu je nutné spustit z kořene repozitáře s nastaveným `PYTHONPATH=src` (nebo projekt nainstalovat jako balíček).
  - macOS/Linux (jednorázově): `PYTHONPATH=src python -m billiard ...`
  - Windows PowerShell: `$Env:PYTHONPATH = 'src'; python -m billiard ...`
  - Windows CMD: `set PYTHONPATH=src && python -m billiard ...`
- Default run (elipsa):
  - `python -m billiard`
- Elipsa, jedna trajektorie s animací:
  - `python -m billiard --shape ellipse --a 5 --ecc 0.6 --single --animate`
- Elipsa, dávka startů a úhlů, uložení Poincaré mapy:
  - `python -m billiard --shape ellipse --a 5 --ecc 0.6 --n-inside 3 --n-outside 3 --n-angles 7 --angle-min 10 --angle-max 170 --bounces 500 --save-poincare out/poincare_ellipse.png`
- Stadion, jedna trajektorie, animace + uložení videa:
  - `python -m billiard --shape stadium --R 2.0 --L 5.0 --single --animate --save-anim out/stadium.mp4`

Argumenty, které lze v CLI zadat:
- `--shape {ellipse,stadium}`: volba tvaru.
- `--single`: spustí jednu trajektorii; bez `--single` se provede dávka různých startů/úhlů.
  - `--x0 --y0 --vx --vy`: počáteční poloha a směr pro single běh.
- Exklusivní pro elipsu:
  - `--a`, `--b` (délka poloos) nebo `--ecc` (excentricita) (pokud je zadáno `--ecc=e`, pak `b = a*sqrt(1-e^2))
  - `--n-inside`, `--n-outside`: počet startů mezi ohnisky, resp. vně ohnisek
- Exklusivní pro stadion: 
  - `--R` (poloměr kružnic), `--L` (polovina délky rovné části).
  - `--n-starts`: počet startů
- Společné: 
  - `--n-angles`: počet různých počátečních úhlů, `--angle-min`, `--angle-max`: rozptyl počátečních úhlů
  - `--speed` (rychlost kuličky)
  - `--bounces` (max. počet odrazů, poté simulace skončí), 
  - `--animate` (toggle na animaci trajektorie, default je bez animace), 
  - `--anim-interval` (ms, fps = 1/anim-interval), 
  - `--save-anim PATH` (GIF/MP4), `--save-poincare PATH` (PNG) (cesta na uložení animace/poincareho mapy, např. .../Media/traj.mp4 uloží traj.mp4 do Media)

Poznámka k ukládání animací: u MP4 může být vyžadován systémový `ffmpeg`. Pokud není dostupný, program se pokusí přepnout na GIF. 
Aby se zrychlil render animace, doporučuje se zvýšit --anim-interval a --speed a mít rozumnou velikost --bounces.

Použití z Pythonu (API):
```python
import numpy as np
from billiard.state import State
from billiard.shapes import EllipseShape
from billiard.simulation import run_shape
from billiard.visualize import animate_trajectory_shape, plot_poincare_shape

# Definice tvaru a počátečního stavu
shape = EllipseShape(a=5.0, b=3.0)
s0 = State(pos=np.array([0.0, 0.0]), dir=np.array([0.2, 0.8]), speed=20.0, time=0.0)

# Výpočet trajektorie
states, bounces = run_shape(s0, shape, max_bounces=300)
print(f"Bounces: {bounces}")

# Animace trajektorie (okno)
animate_trajectory_shape(states, shape, interval_ms=25)

# Uložení animace místo zobrazení (vyžaduje ffmpeg pro MP4)
# animate_trajectory_shape(states, shape, interval_ms=25, save_path="out/ellipse_traj.mp4")

# Poincarého (Birkhoffova) mapa – zobrazit a/nebo uložit PNG
plot_poincare_shape(states, shape, show=True, save_path="out/ellipse_poincare.png")
```

Testy: `pytest -q`

Struktura projektu (zkráceně):
- `src/billiard/` – jádro (geometrie, fyzika, tvary, simulace, vizualizace, CLI)
- `tests/` – jednotkové testy

## Algoritmy a datové struktury

Architektura a rozhraní:
- `Shape` (protokol/rozhraní): definuje `intersect_ray(point, direction)`, `normal_at(point)`, `arc_param(point)`, `draw(ax)`.
- Konkrétní tvary: `EllipseShape(a, b)`, `StadiumShape(R, L)`.
- Stav simulace: `State(pos, dir, speed, time)` (neměnný dataclass).
- Krok fyziky: `advance_with_time_shape(point, direction, speed, shape)` – najde dopad, spočte odraz a čas do dopadu.
- Běh simulace: `run_shape(initial, shape, max_bounces, ...)` – iteruje kroky a vrací seznam stavů.
- Vizualizace: animace trajektorie a Poincarého/Birkhoffovy mapy (`plot_poincare_shape`, `plot_poincare_groups_shape`).
- CLI: `python -m billiard` přeposílá na `billiard.initiate.main()` (argumenty viz Uživatelská část).

Hlavní algoritmy:
- Přímý pohyb: mezi nárazy uniformní pohyb `x(t) = x0 + v̂ * (speed * t)`.
- Zrcadlový odraz: `v' = v - 2 (v·n) n`, kde `n` je vnější normála v bodě dopadu (vše normalizováno).
- Dopad na elipsu: průsečík směrové polopřímky s elipsou `x^2/a^2 + y^2/b^2 = 1` řešením kvadratické rovnice; ošetřeny okrajové stavy (tečna/start na hraně) jemným „nudge“ (postrčením) a "přisnapnutím" výsledku zpět na křivku.
- Dopad na stadion: kandidáty tvoří průsečíky
  - s vodorovnými úseky `y = ±R` pro `x ∈ [-L, L]`,
  - s polokružnicemi poloměru `R` se středy `(-L, 0)` a `(L, 0)`; vybere se nejmenší kladné `t`.
  Při tečných situacích se používá numerické „nudge“.
- Normála: elipsa – normalizovaný gradient impl. funkce; stadion – [0,±1] na rovných částech, na polokružnicích vektor směrem do středu kružnice.
- Poincaré/Birkhoff: `s = arc_param(point) ∈ [0,1)`, `p = -dot(v̂, n_out) = sin(ψ)` (ψ je úhel mezi směrovým vektorem a tečnou). Pro elipsu jsou v dávkách rozlišeny skupiny startů „inside/outside“ dle ohnisek.

Datové struktury a typy:
- `State`: neměnný záznam stavu; `pos: np.ndarray`, `dir: np.ndarray`, `speed: float`, `time: float`.
- Rozhraní `Shape`: umožňuje přidat další tvary bez změny simulátoru.
- Trajektorie: seznam `State`, velikost úměrná počtu odrazů.

Složitost a limity:
- Časová složitost výpočtu jedné trajektorie je O(N) pro N odrazů; každý krok dělá konstantní množství aritmetiky/průsečíků.
- Numerická stabilita je ošetřena tolerancemi a „nudge“; extrémní tečné případy mohou vyžadovat vyšší přesnost.

Testování:
- Jednotkové testy pro geometrii/fyziku/simulaci/stadium (`tests/`), spouštění `pytest -q`.
