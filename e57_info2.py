import pye57, numpy as np, time
p = r"\\GENAS\Public\ARCHIVE2026\132_HMILIT\132_BdTerrain\132_SCAN19112026\Contour.e57"
f = pye57.E57(p)
h = f.get_header(0)
print("bounds", [h.cartesianBounds[k].value() for k in ("xMinimum","xMaximum","yMinimum","yMaximum","zMinimum","zMaximum")])
print("rot", h.rotation, "trans", h.translation)
data, bufs = f.make_buffers(["cartesianX","cartesianY","cartesianZ"], 4_000_000)
rd = h.points.reader(bufs)
t=time.time(); n = rd.read(); print("chunk", n, "in", round(time.time()-t,1),"s")
for k,v in data.items(): print(k, v[:n].min(), v[:n].max())
