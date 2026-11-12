
import matplotlib as mpl

def set_style(style='mnras', Hratio=1.0, Wfrac=1.0):
    if style == 'talk':
      size, fsize = 6, 16
    if style == 'book':
      size, fsize = 5.39, 12
    if style == 'mnras':
      size, fsize = 3.32, 8.1
    if style == 'mnras-fw':
      size, fsize = 6.97, 9.5
    if style == 'fw-9':
      size, fsize = 6.97, 12
    if style == '3p':
       size, fsize = 3.32, 12
    
    mpl.rcParams['figure.figsize'] = (size*Wfrac, (size*Wfrac)*Hratio)
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['font.serif'] = ['Times', 'Liberation Serif', 'Times New Roman']
    mpl.rcParams['font.size'] = fsize
    mpl.rcParams['legend.fontsize'] = 'medium'
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['legend.edgecolor'] = 'k'
    mpl.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams['axes.linewidth'] = 1.0
    
    if style == 'fw-9': mpl.rcParams['axes.linewidth'] = 1.5
    
    try:
      mpl.rcParams['xtick.minor.visible'] = True
      mpl.rcParams['ytick.minor.visible'] = True
    except: pass
    try:
      mpl.rcParams['xtick.top'] = True
      mpl.rcParams['ytick.right'] = True
    except: pass
    try:
      mpl.rcParams['xtick.direction'] = 'in'
      mpl.rcParams['ytick.direction'] = 'in'
    except: pass


