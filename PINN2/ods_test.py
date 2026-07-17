from OceanDataStore import OceanDataCatalog



catalog = OceanDataCatalog(catalog_name="noc-stac")

ds = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/T1m_4d',
                     start_datetime='1976-01',
                     end_datetime='2025-12',
                     variable_names=["thetao_con", "so_abs"],
                    bbox=(-180, 60, 180, 90))


ds_u = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/U1m_4d',
                     start_datetime='1976-01',
                     end_datetime='2025-12',
                     variable_names=["uo"],
                    bbox=(-180, 60, 180, 90))

ds_v = catalog.open_dataset(id='noc-npd-era5/npd-eorca12-era5v1/r1i1c1f1/V1m_4d',
                     start_datetime='1976-01',
                     end_datetime='2025-12',
                     variable_names=["vo"],
                    bbox=(-180, 60, 180, 90))
print(ds)
print(ds_u)
print(ds_v)