# .latexmkrc
$out_dir = 'build';
$aux_dir = 'build'; # Optional, but recommended for clean separation

# \include{slides/...} escribe un .aux por archivo dentro de $aux_dir, de modo
# que build/slides/ debe existir antes de compilar o el primer pase aborta.
system('mkdir', '-p', 'build/slides');
