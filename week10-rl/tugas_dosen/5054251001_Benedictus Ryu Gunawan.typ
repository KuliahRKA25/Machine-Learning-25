#import "@preview/cetz:0.4.2"

#set document(title: "Tugas Q-Learning: Drone Pengantar Obat", author: "Benedictus Ryu Gunawan")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt, lang: "id")
#set par(justify: true)
#set heading(numbering: "1.")
#set figure(gap: 0.6em)
#show figure.caption: set text(size: 9.5pt)

// ---------- palet & helper grid ----------
#let c-start = rgb("#cfe3ff")
#let c-goal = rgb("#c8f0c8")
#let c-obs = rgb("#ffc9c9")
#let c-path = rgb("#fff1b8")
#let c-blank = white

// pusat sel (baris r, kolom c); baris 0 di atas
#let ctr(r, c) = (c + 0.5, 3.5 - r)

#let cell-fill(r, c, path: ()) = {
  if (r, c) == (3, 0) { c-start }
  else if (r, c) == (0, 3) { c-goal }
  else if (r, c) == (1, 2) { c-obs }
  else if path.contains((r, c)) { c-path }
  else { c-blank }
}

#let cell-label(r, c) = {
  if (r, c) == (3, 0) { [*S*] }
  else if (r, c) == (0, 3) { [*G* \ +40] }
  else if (r, c) == (1, 2) { [*X* \ −25] }
  else { [] }
}

// gambar grid 4x4 + elemen tambahan (extra: fungsi yg mengembalikan elemen cetz)
#let grid4(path: (), labels: true, extra: none, size: 1.5cm) = cetz.canvas(length: size, {
  import cetz.draw: *
  for r in range(4) {
    for c in range(4) {
      rect((c, 3 - r), (c + 1, 4 - r), fill: cell-fill(r, c, path: path), stroke: 0.6pt + gray)
      content((c + 0.5, 3.88 - r), text(6pt, fill: gray.darken(30%))[(#r,#c)])
      if labels { content(ctr(r, c), align(center, text(8pt, cell-label(r, c)))) }
    }
  }
  if extra != none { extra }
})

// panah langkah antar sel
#let step(a, b, col: blue, w: 1.6pt) = {
  import cetz.draw: *
  line(ctr(..a), ctr(..b), stroke: w + col, mark: (end: ">", fill: col, scale: 0.7))
}

#let path-arrows(cells, col: blue) = {
  for i in range(cells.len() - 1) { step(cells.at(i), cells.at(i + 1), col: col) }
}

// ---------- judul ----------
#align(center)[
  #text(15pt, weight: "bold")[Tugas Reinforcement Learning: Q-Learning] \
  #text(12pt)[Environment "Drone Pengantar Obat"] \
  #v(4pt)
  #table(
    columns: 2, stroke: 0.5pt, inset: 5pt,
    [*Nama*], [Benedictus Ryu Gunawan],
    [*NRP*], [5054251001],
  )
]

= Ringkasan Environment

Grid $4 times 4$ dengan koordinat (baris, kolom), baris 0 paling atas. Start $(3,0)$, goal $(0,3)$ dengan reward $+40$ (terminal), obstacle $(1,2)$ dengan reward $-25$ (terminal). Langkah biasa bernilai $-1$, menabrak batas area bernilai $-3$ dan drone tetap di petak semula. Parameter: $alpha = 0.4$, $gamma = 0.9$, semua $Q$ awal $= 0$.

$ Q(s,a) <- Q(s,a) + alpha [ r + gamma dot max_(a') Q(s',a') - Q(s,a) ] $

dengan $max Q(s', dot) = 0$ jika $s'$ terminal.

#figure(
  grid4(),
  caption: [Peta environment. *S* = start (gudang farmasi), *G* = goal (IGD), *X* = obstacle.],
)

Q-value yang sudah dipelajari dari Episode 1 dan 2 (selain yang tercantum, nilainya 0):

#figure(
  grid(
    columns: 2, column-gutter: 1.2cm,
    grid4(path: ((3,1), (3,2), (3,3), (2,3), (1,3)), size: 1.2cm,
      extra: path-arrows(((3,0), (3,1), (3,2), (3,3), (2,3), (1,3), (0,3)), col: blue)),
    grid4(path: ((2,0), (2,1), (1,1)), size: 1.2cm, extra: {
      import cetz.draw: *
      path-arrows(((3,0), (2,0)), col: purple)
      // tabrak batas kiri lalu kembali

      path-arrows(((2,0), (2,1), (1,1), (1,2)), col: purple)
    }),
    [Episode 1 (berhasil ke goal)], [Episode 2 (menabrak batas, lalu masuk obstacle)],
  ),
  caption: [Lintasan Episode 1 dan Episode 2.],
)

#block(breakable: false, width: 100%, align(center, grid(
  columns: 2, column-gutter: 1cm,
  table(
    columns: 3, inset: 5pt, stroke: 0.5pt, align: center,
    table.header(table.cell(colspan: 3)[*Episode 1*], [*State*], [*Action*], [*Q*]),
    [(1,3)], [↑], [16],
    [(3,0)], [→], [−0.4],
    [(3,1)], [→], [−0.4],
    [(3,2)], [→], [−0.4],
    [(3,3)], [↑], [−0.4],
    [(2,3)], [↑], [−0.4],
  ),
  table(
    columns: 3, inset: 5pt, stroke: 0.5pt, align: center,
    table.header(table.cell(colspan: 3)[*Episode 2*], [*State*], [*Action*], [*Q*]),
    [(3,0)], [↑], [−0.4],
    [(2,0)], [←], [−1.2],
    [(2,0)], [→], [−0.4],
    [(2,1)], [↑], [−0.4],
    [(1,1)], [→], [−10],
  ),
)))

= Episode 3 (Backward Update)

Path: $(2,2) ->^(→) (2,3) ->^(↑) (1,3) ->^(↑) (0,3)$. Pembaruan dilakukan mundur, dimulai dari langkah terakhir yang mencapai goal, sehingga nilai goal langsung "mengalir" ke state sebelumnya.

#figure(
  grid4(path: ((2,2), (2,3), (1,3)), extra: {
    import cetz.draw: *
    path-arrows(((2,2), (2,3), (1,3), (0,3)), col: blue)
    content((3.78, 3.0), text(9pt, fill: blue.darken(20%))[①])
    content((3.78, 2.0), text(9pt, fill: blue.darken(20%))[②])
    content((3.0, 1.72), text(9pt, fill: blue.darken(20%))[③])
    content((2.5, 1.32), text(7pt)[start])
  }),
  caption: [Lintasan Episode 3. Urutan update backward: ① $Q((1,3),↑)$, ② $Q((2,3),↑)$, ③ $Q((2,2),→)$.],
)

== (a) $Q((1,3), ↑)$

Transisi $(1,3) ->^(↑) (0,3)$: $r = +40$, $s' = (0,3)$ terminal sehingga $max Q(s', dot) = 0$. Nilai lama $Q((1,3),↑) = 16$.

$
  Q((1,3),↑) &= 16 + 0.4 [40 + 0.9 dot 0 - 16] \
  &= 16 + 0.4 dot 24 \
  &= 16 + 9.6 = bold(25.6)
$

== (b) $Q((2,3), ↑)$

Transisi $(2,3) ->^(↑) (1,3)$: $r = -1$. Nilai lama $Q((2,3),↑) = -0.4$. Karena update backward, $max Q((1,3), dot)$ memakai hasil (a):

$ max_(a') Q((1,3), a') = max(25.6, 0, 0, 0) = 25.6 $

$
  Q((2,3),↑) &= -0.4 + 0.4 [-1 + 0.9 dot 25.6 - (-0.4)] \
  &= -0.4 + 0.4 [-1 + 23.04 + 0.4] \
  &= -0.4 + 0.4 dot 22.44 \
  &= -0.4 + 8.976 = bold(8.576)
$

== (c) $Q((2,2), →)$

Transisi $(2,2) ->^(→) (2,3)$: $r = -1$. $Q((2,2),→)$ belum pernah diperbarui, jadi nilai lamanya $0$. Dari hasil (b):

$ max_(a') Q((2,3), a') = max(8.576, 0, 0, 0) = 8.576 $

$
  Q((2,2),→) &= 0 + 0.4 [-1 + 0.9 dot 8.576 - 0] \
  &= 0.4 [-1 + 7.7184] \
  &= 0.4 dot 6.7184 = bold(2.68736) approx bold(2.687)
$

#align(center, table(
  columns: 4, inset: 6pt, stroke: 0.5pt, align: center,
  table.header([*Langkah*], [*Q(s,a)*], [*Sebelum*], [*Sesudah*]),
  [①], [$Q((1,3),↑)$], [16], [*25.6*],
  [②], [$Q((2,3),↑)$], [−0.4], [*8.576*],
  [③], [$Q((2,2),→)$], [0], [*2.687*],
))

= Analisis Policy

Gambar berikut menunjukkan semua Q-value yang bukan nol setelah Episode 3. Panah hijau berarti Q positif (menuju goal), panah merah berarti Q negatif.

#let qarrow(r, c, dir, val) = {
  import cetz.draw: *
  let (x, y) = ctr(r, c)
  let d = (
    "u": ((0, 0.08), (0, 0.42), (0.17, 0.3)),
    "d": ((0, -0.08), (0, -0.42), (0.17, -0.3)),
    "l": ((-0.08, 0), (-0.42, 0), (-0.3, -0.12)),
    "r": ((0.08, 0), (0.42, 0), (0.3, -0.12)),
  ).at(dir)
  let col = if val > 0 { green.darken(25%) } else { red.darken(10%) }
  line((x + d.at(0).at(0), y + d.at(0).at(1)), (x + d.at(1).at(0), y + d.at(1).at(1)),
    stroke: 1.3pt + col, mark: (end: ">", fill: col, scale: 0.55))
  content((x + d.at(2).at(0), y + d.at(2).at(1)), text(6.5pt, fill: col, weight: "bold", str(val)))
}

#figure(
  grid4(labels: false, size: 2.1cm, extra: {
    import cetz.draw: *
    content(ctr(3, 0), text(8pt)[*S*])
    content(ctr(0, 3), text(8pt)[*G*])
    content(ctr(1, 2), text(8pt)[*X*])
    qarrow(3, 0, "r", -0.4)
    qarrow(3, 0, "u", -0.4)
    qarrow(3, 1, "r", -0.4)
    qarrow(3, 2, "r", -0.4)
    qarrow(3, 3, "u", -0.4)
    qarrow(2, 3, "u", 8.576)
    qarrow(1, 3, "u", 25.6)
    qarrow(2, 0, "l", -1.2)
    qarrow(2, 0, "r", -0.4)
    qarrow(2, 1, "u", -0.4)
    qarrow(1, 1, "r", -10)
    qarrow(2, 2, "r", 2.687)
  }),
  caption: [Peta Q-value setelah Episode 3 (action yang tidak digambar bernilai 0).],
)

*Dari state (1,1): hindari action → (kanan).* \
$Q((1,1),→) = -10$, nilai paling kecil di state ini (action lain masih 0). Action → membawa drone ke $(1,2)$, yaitu obstacle: drone mendarat darurat dengan reward $-25$ dan episode berakhir tanpa mencapai goal. Nilai $-10 = 0 + 0.4(-25 + 0 - 0)$ adalah sinyal penalti terbesar di seluruh Q-table.

*Dari state (2,0): hindari action ← (kiri).* \
$Q((2,0),←) = -1.2$, lebih buruk daripada $Q((2,0),→) = -0.4$ dan action lain ($0$). Kolom 0 adalah batas kiri area, jadi ← membuat drone menabrak batas: reward $-3$ dan drone tetap di $(2,0)$. Langkah ini membuang energi dan waktu tanpa kemajuan: $-1.2 = 0 + 0.4(-3 + 0.9 dot 0 - 0)$.

*Dari state (2,3): action terbaik adalah ↑ (atas).* \
$Q((2,3),↑) = 8.576$, satu-satunya nilai positif di state ini (action lain 0). Action ↑ membawa drone ke $(1,3)$, dan dari sana ↑ sekali lagi mencapai goal ($Q((1,3),↑) = 25.6$). Jadi $arg max_a Q((2,3),a) = ↑$, yang berarti hanya 2 langkah lagi ke IGD dan tidak melewati obstacle.

= Jalur Optimal

Jarak Manhattan dari $(3,0)$ ke $(0,3)$ adalah $3 + 3 = 6$ langkah, jadi jalur optimal minimal 6 langkah tanpa menabrak batas dan tanpa melewati $(1,2)$. Jalur yang sesuai dengan Q-value yang sudah dipelajari (Episode 1 dan 3) adalah:

$ (3,0) ->^(→) (3,1) ->^(→) (3,2) ->^(→) (3,3) ->^(↑) (2,3) ->^(↑) (1,3) ->^(↑) (0,3) $

#figure(
  grid4(path: ((3,1), (3,2), (3,3), (2,3), (1,3)), size: 1.7cm, extra: {
    import cetz.draw: *
    path-arrows(((3,0), (3,1), (3,2), (3,3), (2,3), (1,3), (0,3)), col: green.darken(30%))
    let rw = ("−1", "−1", "−1", "−1", "−1", "+40")
    let pos = ((1.0, 0.68), (2.0, 0.68), (3.0, 0.68), (3.8, 1.0), (3.8, 2.0), (3.8, 3.0))
    for i in range(6) { content(pos.at(i), text(7pt, fill: green.darken(40%), weight: "bold", rw.at(i))) }
  }),
  caption: [Jalur optimal $(3,0) -> (0,3)$ beserta reward setiap langkah. Obstacle $(1,2)$ tidak dilewati.],
)

*Total reward (tanpa diskon):*

#align(center, table(
  columns: 4, inset: 5pt, stroke: 0.5pt, align: center,
  table.header([*Langkah*], [*Transisi*], [*Action*], [*Reward*]),
  [1], [(3,0) → (3,1)], [→], [−1],
  [2], [(3,1) → (3,2)], [→], [−1],
  [3], [(3,2) → (3,3)], [→], [−1],
  [4], [(3,3) → (2,3)], [↑], [−1],
  [5], [(2,3) → (1,3)], [↑], [−1],
  [6], [(1,3) → (0,3)], [↑], [+40],
  table.cell(colspan: 3)[*Total*], [*+35*],
))

$ G = 5 times (-1) + 40 = -5 + 40 = bold(35) $

Catatan: langkah terakhir hanya diberi reward goal $+40$ (sesuai Episode 1, di mana $Q((1,3),↑) = 0.4 dot 40 = 16$). Jalur 6 langkah lain yang tidak melewati $(1,2)$, misalnya $↑ ↑ ↑ → → →$ lewat baris 0, juga menghasilkan total $35$. Namun jalur di atas adalah jalur yang dipilih secara greedy oleh Q-table saat ini, karena hanya jalur ini yang sudah memiliki Q positif menuju goal.
