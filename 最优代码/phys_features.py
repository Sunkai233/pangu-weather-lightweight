# -*- coding: utf-8 -*-
"""物理头特征(用户①:物理算好规律部分让网络免学)。文件名YYYYMMDDHH→datetime。
- 太阳辐射cos天顶角(治2m_t昼夜循环,J2000天文公式,=GraphCast/AIFS insolation)
- 时钟sin/cos(年/日周期,治季节日循环)
- sdor次网格地形std(治地形失配,第二轮虽误差均匀但近地面物理相关)
全部0参数、纯几何物理量,网络免学→减负+救温度。"""
import numpy as np
from datetime import datetime

def solar_zenith_cos(dt, lat2d, lon2d):
    """cos(太阳天顶角): 白天>0夜间0。J2000低精度天文公式(误差<0.01°,气象足够)。"""
    d = (dt - datetime(2000, 1, 1, 12)).total_seconds() / 86400.0   # J2000儒略日
    g = np.deg2rad((357.529 + 0.98560028 * d) % 360)                # 平近点角
    q = (280.459 + 0.98564736 * d) % 360                            # 太阳平黄经
    L = np.deg2rad((q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) % 360)  # 黄经
    eps = np.deg2rad(23.439 - 0.00000036 * d)                       # 黄赤交角
    decl = np.arcsin(np.sin(eps) * np.sin(L))                       # 太阳赤纬
    ra = np.arctan2(np.cos(eps) * np.sin(L), np.cos(L))             # 太阳赤经
    gmst = (18.697374558 + 24.06570982441908 * d) % 24             # 格林尼治平恒星时(h)
    lst = np.deg2rad(gmst * 15.0) + np.deg2rad(lon2d)              # 本地恒星时(rad)
    ha = lst - ra                                                   # 时角
    latr = np.deg2rad(lat2d)
    coszen = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(ha)
    return np.clip(coszen, 0.0, None).astype(np.float32)            # 夜间截0

def clock_feats(dt):
    """年/日周期 sin/cos (4个标量,治季节+日循环相位)。"""
    yp = (dt.timetuple().tm_yday - 1) / 365.25
    hp = dt.hour / 24.0
    return np.array([np.sin(2*np.pi*yp), np.cos(2*np.pi*yp), np.sin(2*np.pi*hp), np.cos(2*np.pi*hp)], np.float32)

def sdor_field(topo_m, size=3):
    """次网格地形std(代理AIFS sdor): 3x3邻域标准差,复杂地形大。"""
    try:
        from scipy.ndimage import generic_filter
        return generic_filter(topo_m.astype(np.float32), np.std, size=size).astype(np.float32)
    except Exception:
        gx, gy = np.gradient(topo_m); return np.sqrt(gx**2 + gy**2).astype(np.float32)

if __name__ == "__main__":  # sanity: 太阳cos天顶角应在正午当地最大、夜间0、夏至北半球高
    lat = np.linspace(90, -90, 73)[:, None]; lon = np.linspace(0, 359, 144)[None, :]
    lat2d = np.repeat(lat, 144, 1); lon2d = np.repeat(lon, 73, 0)
    for s in ["2012062112", "2012122112", "2012062100"]:
        dt = datetime.strptime(s, "%Y%m%d%H")
        cz = solar_zenith_cos(dt, lat2d, lon2d)
        # 找最大日照点(应在太阳直射点附近)
        i, j = np.unravel_index(cz.argmax(), cz.shape)
        print(f"{s}: max cos_zen={cz.max():.3f} @lat{lat2d[i,j]:.0f} lon{lon2d[i,j]:.0f}; 全球日照>0占比{(cz>0).mean()*100:.0f}%(应~50%昼半球)", flush=True)
        print(f"        时钟特征={clock_feats(dt)}", flush=True)
    print("[check] 夏至(0621)max应在北纬~23°,冬至(1221)在南纬~23°,昼半球~50%", flush=True)
