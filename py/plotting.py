#####################################
# Styling for all publication plots #
#####################################

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\DeclareSIUnit\year{yr}\usepackage[version=4]{mhchem}')
#plt.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{lmodern}\usepackage[T1]{fontenc}\DeclareUnicodeCharacter{2212}{-}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\DeclareSIUnit\year{yr}\usepackage[version=4]{mhchem}')
plt.rc('text.latex', preamble=r'\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\DeclareSIUnit\year{yr}\usepackage[version=4]{mhchem}')
plt.rc('pgf', texsystem='pdflatex', preamble=r'\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\DeclareSIUnit\year{yr}\usepackage[version=4]{mhchem}')
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
mnras_width = 3.38 # inches
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

## Specific plotting functions
def plot_m_a(m, a, label='', bfit=None, xlims=[5,25], ylims=[0,1]):
    phi = np.linspace(0, 2*np.pi, 100)
    plt.figure(figsize=(mnras_width, 3))
    plt.plot(m, a, 'o', ms=1, c='0.6', rasterized=True, label=r"Samples")
    m_mean, a_mean = np.mean(m), np.mean(a)
    m_std, a_std = np.std(m), np.std(a)
    cov = np.cov((m,a))
    l, c = np.linalg.eig(cov)
    if (bfit != None):
        plt.plot(bfit[0], bfit[1], ms=5, c='k', marker='*')
    print("m = {:.2f} +/- {:.2f}, a = {:.2f} +/- {:.2f}".format(m_mean, m_std, a_mean, a_std))
    xp = np.array([2*c@np.diag(np.sqrt(l))@v for v in [np.array([np.cos(x), np.sin(x)]).T for x in phi]])
    plt.plot(m_mean + 2*m_std*np.cos(phi), a_mean + 2*a_std*np.sin(phi), 'b--', label=r"Uncorrelated Gaussian")
    plt.plot(m_mean + xp[:,0], a_mean + xp[:,1], 'r-', label=r"Full Gaussian")
    plt.plot(m_mean+2*m_std*np.array([1,1,-1,-1,1]), a_mean+2*a_std*np.array([1,-1,-1,1,1]), 'k:', label=r"`Box method'")
    plt.ylim(ylims)
    plt.xlim(xlims)
    plt.xlabel(r"Black hole mass [$M_\odot$]")
    plt.ylabel(r"Black hole spin $a_\ast$")
    plt.legend(frameon=False)
    if label !='':
        plt.tight_layout()
        plt.savefig("figures/"+label+".pdf", backend='pgf')
    else:
        plt.text(0.95*xlims[1], 0.1, label, ha='right', va='center')
    plt.show()