# Switched coordinates problem — agent handoff

This repo uses **three different coordinate conventions** in different places. They are easy to mix up because Shapely’s `(x, y)` is not always `(longitude, latitude)`.

---

## Three conventions in play

### 1. ETAS inversion / domain polygons — Shapely `(x, y) = (latitude, longitude)`

- `shape_coords` and `california_shape.npy` store vertices as **`[lat, lon]`** per row.
- `Polygon(shape_coords)` → Shapely coords are **`(x=lat, y=lon)`**.
- `etas.inversion.polygon_surface` assumes this: it uses `bounds[0]` and `bounds[2]` as latitudes for the Albers projection.
- `etas.data_utils.latlon_rectangular_grid_in_polygon` documents this explicitly: *“Shapely `(x, y) = (lat, lon)`”*; `Point(lat, lon)` for containment checks.

### 2. Catalog / history / classic ETAS event dict — **`x = longitude`, `y = latitude`**

Example from the thinning notebooks:

```python
history_event_x = -120   # lon
history_event_y = 38     # lat
history_dict = {"m": ..., "x": history_event_x, "y": history_event_y, "t": ...}
```

DataFrames use named columns `latitude` / `longitude`. Matplotlib maps use **`scatter(lon, lat)`** → `(x, y) = (lon, lat)`.

### 3. Kernel helpers in notebooks — **`a_h(x, y, ...)` with `(x, y) = (lon, lat)`**

`a_h` passes haversine as `(lat, lon)` by swapping:

```python
spatial_distance_squared_km2(y, x, H["y"], H["x"])  # (lat, lon, lat_H, lon_H)
```

So **`H["x"]` is lon, `H["y"]` is lat**, but function args **`x`/`y` are also lon/lat**, not Shapely vertex order.

---

## Where the swap happens (and why it breaks)

| Location | What it assumes | Actual data |
|----------|-----------------|-------------|
| `A_h(poly, ...)` docstring | Poly vertices **`(lon, lat)`** | Circular domain built as **`(lat, lon)`** |
| `A_h_for_weights(poly, ...)` | Poly vertices **`(lat, lon)`** | Matches `circle` from `regular_polygon` |
| `regular_polygon(cx, cy, ...)` | Called as `regular_polygon(history_event_y, history_event_x, ...)` → **`(cx=lat, cy=lon)`** | Produces Shapely polygon in inversion convention |
| Plotting domain | `lat_poly, lon_poly = coords[:,0], coords[:,1]` then `ax.plot(lon_poly, lat_poly)` | Swaps from `(lat,lon)` storage to map `(lon,lat)` |
| `get_ordered_voronoi_cells` | Input polygon in `(lat, lon)`; **`transform(_swap_xy, ...)`** → `(lon, lat)` for Voronoi | Explicit boundary conversion |
| `california_lonlat = transform(_swap_xy, circle)` | Same pattern before lon/lat-only code paths | Explicit boundary conversion |

**Core bug:** The generic `A_h` integrator treats polygon bounds as **`(lon, lat)` on the grid axes**, but the circular study region (and California `shape_coords` polygons) are stored as **`(lat, lon)`**. That silently mislabels axes in the quadrature grid and breaks spatial integrals unless you use the fixed variant.

That is why **`A_h_for_weights`** exists in `single_point_history_circular_domain_thinning.ipynb` — a duplicate integral with the correct `(lat, lon)` grid, used for thinning weights so history/background weights sum to 1.

---

## `_swap_xy` role

```python
def _swap_xy(x, y):
    return y, x
```

Used at **convention boundaries**:

- Voronoi: polygon stored as `(lat, lon)` → swap to standard geo `(lon, lat)` before `voronoi_diagram` and `Point(lon, lat)`.
- Some analytical paths: `circle` → `california_lonlat` before code that expects map-style ordering.

It is **not** a universal fix; each function still needs a documented convention.

---

## Safe rules for new code

1. **Pick one convention per function** and state it in the docstring.
2. **`spatial_distance_squared_km2(lat, lon, lat_k, lon_k)`** — always **`(lat, lon)`** order (see `etas/utility_functions.py`).
3. **Inversion polygons / grid code:** Shapely **`(x, y) = (lat, lon)`**; use `Point(lat, lon)`.
4. **Maps / scatter:** matplotlib **`(x, y) = (lon, lat)`**.
5. **`history_dict` / ETAS event dicts:** **`x=lon, y=lat`** — do not assume `Point.x` is lon without checking how the polygon was built.
6. **At boundaries:** convert explicitly (`_swap_xy`, or index swap `coords[:,1], coords[:,0]` for plots).
7. **Do not reuse `A_h`** for `(lat, lon)` polygons without fixing it or using `A_h_for_weights`-style logic.

---

## Related refactors (grid vs classic)

- Classic ETAS uses haversine / km-per-degree at latitude for spatial distances.
- Grid continuation was refactored from UTM Euclidean offsets to **lat/lon + haversine km²** (`forecast_intensity.py`, `grid_simulation.py`, `data_utils.py`) to match classic.
- **`rate_computation.py`** may still use the older UTM GPU path — not fully aligned.
- Notebook inline copies of `run_etas_per_grid_point_inversion` may lag behind `etas/grid_simulation.py`.

---

## Files most affected

| File | Issue |
|------|--------|
| `notebooks/single_point_history*.ipynb` | Mixed conventions; old `a_h` used degree² `(x-H['x'])**2` without haversine |
| `notebooks/single_point_history_circular_domain_thinning.ipynb` | `A_h` vs `A_h_for_weights` split; plotting swaps; `_swap_xy` for Voronoi/analytical |
| `runnable_code/MAGNET_ETAS_pipeline.py` | Documents `shape_coords` as `[[lat,lon]]` vs Shapely `(lon,lat)` when building from convex hull |
| `etas/data_utils.py` | Canonical `(lat, lon)` Shapely convention for grids |
| `etas/inversion.py` | `polygon_surface`, `haversine` — lat-first argument order |

---

## Quick sanity checks

- History at **(-120°, 38°)** → `H["x"]=-120`, `H["y"]=38`.
- Circle around it: `regular_polygon(38, -120, radius_deg, ...)` → polygon coords ~`(38±r, -120±r)` i.e. **`(lat, lon)`**.
- Plot: `ax.plot(lons, lats)` not `ax.plot(lats, lons)`.
- Haversine call: `spatial_distance_squared_km2(event_lat, event_lon, H["y"], H["x"])`.

---

No repo-wide cleanup has been done; conventions are duplicated across notebooks and modules, so **always read the local docstring before reusing a helper**.
