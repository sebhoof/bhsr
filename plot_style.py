#####################################
# Styling for all publication plots #
#####################################

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\DeclareSIUnit\year{yr}\usepackage[version=4]{mhchem}')
plt.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\DeclareUnicodeCharacter{2212}{-}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\DeclareSIUnit\year{yr}\usepackage[version=4]{mhchem}')
plt.rc('font', **{'family':'serif','size':10})
plt.rc('axes', labelsize=10)
plt.rc('xtick', **{'labelsize':8, 'major.pad':4})
plt.rc('ytick', **{'labelsize':8, 'major.pad':4})
plt.rc('legend', **{'fontsize':8, 'title_fontsize':8})
plt.rc('figure', titlesize=10, figsize=(3.375, 3.))
plt.rc("xtick", direction='in', bottom=True, top=True)
plt.rc("ytick", direction='in', left=True, right=True)
plt.tight_layout.__defaults__ = (0.15, None, None, None)
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.frameon"] = False

jcap_width = 6 # inches
golden_ratio = 0.5*(1.0 + np.sqrt(5.0))

def set_style(gs=10, lts=10, lfs=8, lbls=10, tls=10):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
    plt.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{grffile}\DeclareUnicodeCharacter{2212}{-}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}')
    plt.rc('font', **{'family':'serif', 'size': gs})
    plt.rc('axes', labelsize=lbls)
    plt.rc('xtick', **{'labelsize': tls, 'major.pad': 0.5*tls})
    plt.rc('ytick', **{'labelsize': tls, 'major.pad': 0.5*tls})
    plt.rc('legend', **{'fontsize': lfs, 'title_fontsize': lts})
    plt.rc('figure', titlesize=12)
    plt.rc('mathtext', default='regular')

def log10_special_formatter(x, pos, offset=0, is_xaxis=True):
    res = "$10^{%g}$" % (x+offset)
    if np.abs(x+offset) < 3:
        if is_xaxis:
            res = "$%g^{\phantom{0}}$" % (10.0**(x+offset))
        else:
            res = "$%g$" % (10.0**(x+offset))
    return res

def log10_special_formatter_every_n(x, pos, n=2, is_xaxis=False):
    res = ""
    if x%n == 0:
        res = "$10^{{{:g}}}$".format(x)
        if np.abs(x) < 3:
            res = "${:g}$".format(10.0**x)
            if is_xaxis:
                res += "$^{\phantom{0}}$"
    return res

def pow10_formatter(x, pos, every_n=1, is_xaxis=False):
    lgx = int(np.log10(x))
    if lgx%every_n == 0:
        if np.abs(lgx) < 3:
            if is_xaxis:
                return "${%g}^{\phantom{0}}$" % (x)
            else:
                return "${%g}$" % (x)
        return "$10^{%g}$" % (lgx)

def set_axis_formatter(ax, axxrange=[], axyrange=[], every_n_x=1, every_n_y=1):
    formatter_x = plt.FuncFormatter(lambda x, pos: log10_special_formatter_every_n(x, pos, every_n_x, True))
    formatter_y = plt.FuncFormatter(lambda x, pos: log10_special_formatter_every_n(x, pos, every_n_y))
    if len(axxrange) > 0:
        major_locator = plt.FixedLocator(axxrange)
        minor_locator = plt.FixedLocator([i+j for i in np.log10(range(2, 10)) for j in axxrange])
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_major_formatter(formatter_x)
    if len(axyrange) > 0:
        major_locator = plt.FixedLocator(axyrange)
        minor_locator = plt.FixedLocator([i+j for i in np.log10(range(2, 10)) for j in axyrange])
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_major_formatter(formatter_y)

def axis_limits(a):
    return [np.min(a),np.max(a)]

def sci_format(x):
    sign = np.sign(x)
    lgx = np.log10(np.abs(x))
    exp = int(lgx)
    lead = 10**(lgx-exp)
    return '{:.2f} \\times 10^{{ {:d} }}'.format(sign*lead, exp)
