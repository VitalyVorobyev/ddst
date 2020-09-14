""" Plotting tools """

from .params import DalitzNBins, mdn, mpip, mdp, mpin

import numpy as np

def put_plot_on_axis(figax, data, saveas=None, **kwargs):
    """ """
    fig, ax = figax
    for x, y, lbl, pdict in data:
        ax.plot(x, y, label=lbl, **pdict)

    ax.set(**kwargs)
    ax.legend(loc='best', fontsize=18)
    ax.grid()
    fig.tight_layout()

    if saveas:
        for ext in ['png', 'pdf']:
            fig.savefig(f'plots/{saveas}.{ext}')


#### Common ####
def dd_plot(ax, pdf, lbl, sqrt=True):
    """ D0D{0,+} kinetic energy spectrum
    Args:
        ax - axes
        pdf - DnDpGam, DnDpPin, or DnDnPip object
    """
    nbins=DalitzNBins
    bins = pdf.linspaceAB(nbins)
    mdd = pdf.mddspec(b1=nbins, b2=nbins)
    if sqrt:
        mdd = mdd*2*np.sqrt(bins)
        bins = (np.sqrt(bins) - pdf.da[0] - pdf.da[1])*10**3
        lbl = fr'$E({lbl})$, MeV'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdd)), xlim=(0, 8))
    else:
        lbl = fr'$m^2({lbl})$'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdd)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdd)


#### D0 D+ gamma ####
def dga_dga_plot(ax, pdf, logplot=True):
    """ [D0 gamma] [D+ gamma] Dalitz plot
    Args:
        ax - axes
        pdf - DnDpGam object
    """
    (mdnga, mdpga), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(mdnga, mdpga)
    z, mask = pdf(mdd, mdnga)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnga, mdpga, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^+\gamma)$', xlabel=r'$m(D^0\gamma)$')
    ax.contour(mdnga, mdpga, phsp, levels=1)

def dd_dga_plot(ax, pdf, logplot=True):
    """ D0D+ kinetic energy spectrum
    Args:
        ax - axes
        pdf - DnDpGam object
    """
    (mdd, mdnga), _ = pdf.mgridABAC(500)
    z, mask = pdf(mdd, mdnga)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnga, mdd, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^0D^+)$', xlabel=r'$m(D^0\gamma)$')
    ax.contour(mdnga, mdd, phsp, levels=1)


def dnga_plot(ax, pdf, sqrt=True):
    """ D0 gamma mass spectrum
    Args:
        ax - axes
        pdf - DnDpGam object
    """
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdnga = pdf.mdngaspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdnga *= 2*bins
        lbl = r'$m(D^0\gamma)$'
    else:
        lbl = r'$m^2(D^0\gamma)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdnga)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdnga)

def dpga_plot(ax, pdf, sqrt=True):
    """ D+ gamma mass spectrum
    Args:
        ax - axes
        pdf - DnDpGam object
    """
    nbins=DalitzNBins
    bins = pdf.linspaceBC(nbins)
    mdpga = pdf.mdpgaspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdpga *= 2*bins
        lbl = r'$m(D^+\gamma)$'
    else:
        lbl = r'$m^2(D^+\gamma)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpga)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpga)

#### D0 D0 pi+ ####
def dnpi_dnpi_plot(ax, pdf, logplot=True):
    (md1pi, md2pi), _ = pdf.mgridACBC(512)
    mdd = pdf.mZsq(md1pi, md2pi)
    z, mask = pdf(mdd, md1pi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(md1pi, md2pi, z, cmap=None, levels=100)
    ax.set(
        xlabel=r'$m(D^0_{(1)}\pi^+)$',
        ylabel=r'$m(D^0_{(2)}\pi^+)$',
        xlim=((mdn + mpip)**2, 4.050),
        ylim=((mdn + mpip)**2, 4.050),
    )
    ax.contour(md1pi, md2pi, phsp, levels=1)

def dd_dpi_plot(ax, pdf, logplot=True):
    (mdd, md1pi), _ = pdf.mgridABAC(500)
    z, mask = pdf(mdd, md1pi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(md1pi, mdd, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^0D^0)$', xlabel=r'$m(D^0\pi^+)$')
    ax.contour(md1pi, mdd, phsp, levels=1)

def dpi_lo_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdpi = pdf.mdpilspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdpi *= 2*bins
        lbl = r'$m(D^0\pi^+)$ low'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(2.005, 2.013))
    else:
        lbl = r'$m^2(D^0\pi^+)$ low'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpi)

def dpi_hi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceBC(nbins)
    mdpi = pdf.mdpihspec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdpi *= 2*bins
        lbl = r'$m(D^0\pi^+)$ high'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(2.005, 2.013))
    else:
        lbl = r'$m^2(D^0\pi^+)$ high'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpi)

#### D0 D+ pi0 ###
def dnpi_dppi_plot(ax, pdf, logplot=True):
    (mdnpi, mdppi), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(mdnpi, mdppi)
    z, mask = pdf(mdd, mdnpi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnpi, mdppi, z, cmap=None, levels=100)
    ax.set(
        xlabel=r'$m(D^0\pi^0)$',
        ylabel=r'$m(D^+\pi^0)$',
        xlim=((mdn + mpin)**2, 4.030),
        ylim=((mdp + mpin)**2, 4.050),
    )
    ax.contour(mdnpi, mdppi, phsp, levels=1)

def dnpi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdnpi = pdf.mdnpispec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdnpi *= 2*bins
        lbl = r'$m(D^0\pi^0)$'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdnpi)), xlim=(2.001, 2.008))
    else:
        lbl = r'$m^2(D^0\pi^0)$'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdnpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdnpi)


def dppi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceBC(nbins)
    mdppi = pdf.mdppispec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdppi *= 2*bins
        lbl = r'$m(D^+\pi^0)$'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdppi)), xlim=(2.005, 2.013))
    else:
        lbl = r'$m^2(D^+\pi^0)$'
        ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdppi)),
               xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdppi)

def draw_1d_projections(ax, e, pd, mdpi, bins=250, weights=None, alpha=None,
                        density=False, lims=None, label=None):
    """ Plot 1D projections fiven 3D toy MC events """
    if not lims:
        lims = [(-3, 10), (0, 150), (2004, 2016)]
    labels = (r'$E (MeV)$', r'$p(D^0)$ (MeV)', r'$m(D^0pi^+)$ (MeV)')
    for idx, data in enumerate([e, pd, mdpi]):
        ax[idx].hist(data, bins=bins, weights=weights, alpha=alpha,
                     density=density, range=lims[idx], label=label)
        if label:
            ax[idx].legend()
        ax[idx].grid()
        ax[idx].set_xlabel(labels[idx])

def draw_pdf_projections(ax, x, y, label=None):
    """ Plot 1D projection of 3D PDF given it's values for a
        regular 3D meshgrid """
    labels = (r'$E (MeV)$', r'$p(D^0)$ (MeV)', r'$m(D^0pi^+)$ (MeV)')
    dx = [item[1] - item[0] for item in x]
    projs = [y.sum(axis=atup) * dx[atup[0]] * dx[atup[1]]
        for atup in [(1,2), (0,2), (0,1)]]
    for idx, (lbl, xi, pro) in enumerate(zip(labels, x, projs)):
        ax[idx].plot(xi, pro, label=label)
        if label:
            ax[idx].legend()
        ax[idx].grid()
        ax[idx].set_xlabel(lbl)
