""" Plotting tools """

from .params import DalitzNBins

#### Common ####
def dd_plot(ax, pdf, sqrt=True):
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
        lbl = r'$E(D^0D^+)$, MeV'
    else:
        lbl = r'$m^2(D^0D^+)$'
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
def dpi_dpi_plot(ax, pdf, logplot=True):
    (md1pi, md2pi), _ = pdf.mgridACBC(512)
    mdd = pdf.mZsq(md1pi, md2pi)
    z, mask = pdf(mdd, md1pi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(md1pi, md2pi, z, cmap=None, levels=100)
    ax.set(xlabel=r'$m(D^0_{(1)}\pi^+)$', ylabel=r'$m(D^0_{(2)}\pi^+)$')
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
    else:
        lbl = r'$m^2(D^0\pi^+)$ high'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdpi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdpi)

#### D0 D+ pi0 ###
def dpi_dpi_plot(ax, pdf, logplot=True):
    (mdnpi, mdppi), _ = pdf.mgridACBC(500)
    mdd = pdf.mZsq(mdnpi, mdppi)
    z, mask = pdf(mdd, mdnpi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnpi, mdppi, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^+\pi^0)$', xlabel=r'$m(D^0\pi^0)$')
    ax.contour(mdnpi, mdppi, phsp, levels=1)


def dd_dpi_plot(ax, pdf, logplot=True):
    (mdd, mdnpi), _ = pdf.mgridABAC(500)
    z, mask = pdf(mdd, mdnpi)
    if logplot:
        z = np.log(z+1.000000001)
    phsp = np.zeros(mdd.shape)
    phsp[mask] = 1
    ax.contourf(mdnpi, mdd, z, cmap=None, levels=100)
    ax.set(ylabel=r'$m(D^0D^+)$', xlabel=r'$m(D^0\pi^0)$')
    ax.contour(mdnpi, mdd, phsp, levels=1)


def dnpi_plot(ax, pdf, sqrt=True):
    nbins=DalitzNBins
    bins = pdf.linspaceAC(nbins)
    mdnpi = pdf.mdnpispec(b1=nbins, b2=nbins)
    if sqrt:
        bins = np.sqrt(bins)
        mdnpi *= 2*bins
        lbl = r'$m(D^0\pi^0)$'
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
    else:
        lbl = r'$m^2(D^+\pi^0)$'
    ax.set(xlabel=lbl, ylim=(0, 1.01*np.max(mdppi)), xlim=(bins[0], bins[-1]))
    ax.grid()
    ax.plot(bins, mdppi)
