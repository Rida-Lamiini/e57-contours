import pye57, numpy as np
f = pye57.E57(r"\\GENAS\Public\ARCHIVE2026\132_HMILIT\132_BdTerrain\132_SCAN19112026\Contour.e57")
n = f.scan_count
print("scans:", n)
for i in range(n):
    h = f.get_header(i)
    print(i, "points:", h.point_count, "fields:", h.point_fields)
    try:
        print("  bounds:", [(h.cartesianBounds[k]) for k in ("xMinimum","xMaximum","yMinimum","yMaximum","zMinimum","zMaximum")])
    except Exception as e:
        print("  no bounds", e)
