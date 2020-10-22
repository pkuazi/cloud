import osr

def epsg2wkt(in_proj):     
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.SetFromUserInput(in_proj)
    wkt = inSpatialRef.ExportToWkt()
    return wkt