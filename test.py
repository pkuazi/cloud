import gdal
from datetime import datetime
def gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE):
    xoff_list = []
    yoff_list = []
    
    cnum_tile = int((xsize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    for j in range(cnum_tile + 1):
        xoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * j      
            # the last column                 
        if j == cnum_tile:
            xoff = xsize - BLOCK_SIZE
            # the last row
        xoff_list.append(xoff)
        
    rnum_tile = int((ysize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    for i in range(rnum_tile + 1):
        yoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * i
        if i == rnum_tile:
            yoff = ysize - BLOCK_SIZE
        yoff_list.append(yoff)
    
    if xoff_list[-1]==xoff_list[-2]:
        xoff_list.pop()
    if yoff_list[-1]==yoff_list[-2]:# the last tile overlap with the last second tile
        yoff_list.pop()
        
    return xoff_list, yoff_list

import osr

def epsg2wkt(in_proj):     
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.SetFromUserInput(in_proj)
    wkt = inSpatialRef.ExportToWkt()
    return wkt

def batch_read_test():
    import math
    import numpy as np
    import itertools
    out_proj = epsg2wkt('EPSG:4326')
    BLOCK_SIZE=256
    OVERLAP_SIZE=0
    batch_size = 8
    
    file = '/tmp/test1.tif'
    ds = gdal.Open(file)
    gt = list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize = ds.RasterYSize
    xoff_list,yoff_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
    off_list = [d for d in itertools.product(yoff_list,xoff_list)]

    tile_num= len(off_list)
    batch_num=math.ceil(tile_num/batch_size)
    
    s=datetime.now()
    i = 0
    batch_off=off_list[i*batch_size:(i+1)*batch_size]
    np_off = np.array(batch_off)
    yoff_min = int(np_off[:,0].min())
    yoff_win = int(np_off[:,0].max()+BLOCK_SIZE)-yoff_min
#     print(0,yoff_min,xsize, yoff_win)
    
    data = ds.ReadAsArray(0,yoff_min,xsize, yoff_win)
    
    for j in range(batch_size):
        yoff,xoff = batch_off[j]
        tile_yoff = yoff-yoff_min
        tile_xoff = xoff
        tile = data[:,tile_yoff:tile_yoff+BLOCK_SIZE,tile_xoff:tile_xoff+BLOCK_SIZE]
    print('one batch-wise time:',(datetime.now()-s))
#     for i in range(batch_num):
#         batch_off=off_list[i*batch_size:(i+1)*batch_size]
#         np_off = np.array(batch_off)
#         yoff_min = int(np_off[:,0].min())
#         yoff_win = int(np_off[:,0].max()+BLOCK_SIZE)-yoff_min
#         print(0,yoff_min,xsize, yoff_win)
#         
#         data = ds.ReadAsArray(0,yoff_min,xsize, yoff_win)
#         
#         for j in range(batch_size):
#             yoff,xoff = batch_off[j]
#             tile_yoff = yoff-yoff_min
#             tile_xoff = xoff
#             tile = data[:,tile_yoff:tile_yoff+BLOCK_SIZE,tile_xoff:tile_xoff+BLOCK_SIZE]
            
#             tile_gt = [gt[0]+tile_xoff*gt[1],gt[1],gt[2],gt[3]+yoff*gt[5],gt[4],gt[5]]
#             dst_format = 'GTiff'
#             driver = gdal.GetDriverByName(dst_format)
#             dst_ds = driver.Create('/tmp/test_%s_%s.tif'%(i,j), BLOCK_SIZE, BLOCK_SIZE,  ds.RasterCount, gdal.GDT_Float32)
#             dst_ds.SetGeoTransform(tile_gt)
#             dst_ds.SetProjection(out_proj)
#               
#             if ds.RasterCount == 1:
#                 dst_ds.GetRasterBand(1).WriteArray(tile)
#             else:
#                 for b in range(ds.RasterCount):
#                     dst_ds.GetRasterBand(b + 1).WriteArray(tile[b, :, :])
#             del dst_ds

#     print('batch-wise time:',(datetime.now()-s))

def tile_read_test():
    BLOCK_SIZE=256
    OVERLAP_SIZE=0
    batch_size = 8
    file = '/tmp/test1.tif'
    ds = gdal.Open(file)
    gt = list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize = ds.RasterYSize
    xoff_list,yoff_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
    
      #依次读取切片
    s=datetime.now()
    for i in range(batch_size):
#         print(xoff_list[i%len(xoff_list)], yoff_list[int(i/len(xoff_list))])
        tile = ds.ReadAsArray(xoff_list[i%len(xoff_list)],yoff_list[int(i/len(yoff_list))],BLOCK_SIZE,BLOCK_SIZE)
    print('tile file time:',(datetime.now()-s))
def parallel_batch():
    from multiprocessing.dummy import Pool as ThreadPool  
    
    import math
    import numpy as np
    import itertools
    out_proj = epsg2wkt('EPSG:4326')
    BLOCK_SIZE=256
    OVERLAP_SIZE=0
    batch_size = 8
    
    file = '/tmp/test1.tif'
    ds = gdal.Open(file)
    gt = list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize = ds.RasterYSize
    xoff_list,yoff_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
    off_list = [d for d in itertools.product(yoff_list,xoff_list)]

    tile_num= len(off_list)
    batch_num=math.ceil(tile_num/batch_size)
    
    s=datetime.now()
    i = 0
    batch_off=off_list[i*batch_size:(i+1)*batch_size]
    np_off = np.array(batch_off)
    yoff_min = int(np_off[:,0].min())
    yoff_win = int(np_off[:,0].max()+BLOCK_SIZE)-yoff_min
#     print(0,yoff_min,xsize, yoff_win)
    
    data = ds.ReadAsArray(0,yoff_min,xsize, yoff_win)
    def process(item):
        yoff,xoff = batch_off[item]
        tile_yoff = yoff-yoff_min
        tile_xoff = xoff
        tile = data[:,tile_yoff:tile_yoff+BLOCK_SIZE,tile_xoff:tile_xoff+BLOCK_SIZE]
    items = list()  
    items = range(batch_size)
    pool = ThreadPool()
    pool.map(process, items)
    print('parallel batch-wise time:',(datetime.now()-s))
    pool.close()
    pool.join()
        
def threadparalle_tiles(): 
    from multiprocessing.dummy import Pool as ThreadPool 
    BLOCK_SIZE=256
    OVERLAP_SIZE=0
    batch_size = 8
    file = '/tmp/test1.tif'
    ds = gdal.Open(file)
    gt = list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize = ds.RasterYSize
    xoff_list,yoff_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
    
    def process(item):
        tile = ds.ReadAsArray(xoff_list[item%len(xoff_list)],yoff_list[int(item/len(yoff_list))],BLOCK_SIZE,BLOCK_SIZE)
    
    s=datetime.now()
    items = range(batch_size)
    pool = ThreadPool()
    pool.map(process, items)
    print('parallel tile time:',(datetime.now()-s))
    pool.close()
    pool.join()

 
if __name__ == '__main__':
    parallel_batch()
    file = '/tmp/test1.tif'
    ds = gdal.Open(file)
    xsize=ds.RasterXSize
    ysize = ds.RasterYSize
    xoff_list,yoff_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
    print(xoff_list, yoff_list)
    
    #依次读取切片
    s=datetime.now()
    for i in range(batch_size):
        print(xoff_list[i%len(xoff_list)], yoff_list[int(i/len(xoff_list))])
        tile = ds.ReadAsArray(xoff_list[i%len(xoff_list)],yoff_list[int(i/len(yoff_list))],BLOCK_SIZE,BLOCK_SIZE)
    print('time:',(datetime.now()-s))
    
    #读取整个batch所在整行数据，然后读取对应切片所在位置
    s=datetime.now()
    ystart=0
    if batch_size%len(xoff_list)==0:
        num_row = int(batch_size/len(xoff_list))
    else:
        num_row = int(batch_size/len(xoff_list))+1
    data = ds.ReadAsArray(0,yoff_list[ystart],xsize, BLOCK_SIZE*num_row)
    for i in range(batch_size):
        yoff = ystart+int(i/len(xoff_list))
        print(yoff_list[yoff], xoff_list[i%len(xoff_list)])
        tile=data[yoff_list[yoff]:yoff_list[yoff]+BLOCK_SIZE,xoff_list[i%len(xoff_list)]:xoff_list[i%len(xoff_list)]+BLOCK_SIZE]
    print('time:',(datetime.now()-s))
#     batch_read_test()
    
#     tile_read_test()
#     import math
#     out_proj = epsg2wkt('EPSG:4326')
#     BLOCK_SIZE=256
#     OVERLAP_SIZE=0
#     batch_size = 8
#     
#     file = '/tmp/test1.tif'
#     ds = gdal.Open(file)
#     gt = list(ds.GetGeoTransform())
#     xsize=ds.RasterXSize
#     ysize = ds.RasterYSize
#     xoff_list,yoff_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
#     
#       #依次读取切片
#     s=datetime.now()
#     for i in range(batch_size):
#         print(xoff_list[i%len(xoff_list)], yoff_list[int(i/len(xoff_list))])
#         tile = ds.ReadAsArray(xoff_list[i%len(xoff_list)],yoff_list[int(i/len(yoff_list))],BLOCK_SIZE,BLOCK_SIZE)
#     print('tile file time:',(datetime.now()-s))
    

    
#     #读取整个batch所在整行数据，然后读取对应切片所在位置
#     s=datetime.now()
#     batch_idx=0
#     ystart=batch_idx
# #     if batch_size%len(xoff_list)==0:
# #         num_row = int(batch_size/len(xoff_list))
# #     else:
# #         num_row = int(batch_size/len(xoff_list))+1
# #     data = ds.ReadAsArray(0,yoff_list[ystart],xsize, BLOCK_SIZE*num_row)
#     data = ds.ReadAsArray(0,yoff_list[ystart],xsize, yoff_list[ystart+1])
# #     batch_loc = [[0,0],[batch_size%len(xoff_list), int(batch_size/len(xoff_list))],[batch_size*2%len(xoff_list),int(batch_size*2/len(xoff_list))]]
#     batch_num = int((len(xoff_list)-1)*(len(yoff_list)-1)/batch_size)+1
#     print(batch_num)
#     batch_loc=[]
#     for i in range(batch_num):
#         loc=[batch_size*i%(len(xoff_list)-1), int(batch_size*i/(len(xoff_list)-1))]
#         batch_loc.append([xoff_list[loc[0]],yoff_list[loc[1]]])
#     last_loc = [(len(xoff_list)-1)-(batch_size*i%(len(xoff_list)-1)),(len(yoff_list)-1)-int(batch_size*i/(len(xoff_list)-1))-1]
#     batch_loc[-1]=[xoff_list[last_loc[0]],yoff_list[last_loc[1]]]
#     print(batch_loc)
# #     offlist_idx=[batch_size*n%(len(xoff_list)-1), int(batch_size*n/(len(xoff_list)-1))]
#     xoff_start=xoff_list[0]
#     yoff_start=yoff_list[0]
#     tile_xstart=0
#     tile_ystart=0
#     for n in range(batch_num):
#         tile_xend=batch_size*(n+1)%(len(xoff_list)-1)
#         tile_yend = int(batch_size*(n+1)/(len(xoff_list)-1))+1
# #         offlist_idx=[batch_size*(n+1)%(len(xoff_list)-1), int(batch_size*(n+1)/(len(xoff_list)-1))+1]
# #         xoff_end = xoff_list[offlist_idx[0]]
# #         yoff_end = yoff_list[offlist_idx[1]]
#         data = ds.ReadAsArray(0,yoff_list[tile_ystart],xsize, yoff_list[tile_yend])
#         for t in range(batch_size):
#             for y in range(tile_ystart,tile_yend):
# 
#                 for x in range(tile_xstart,tile_xend):
#                     print(xoff_list[x],yoff_list[y])
#                     tile=data[yoff_list[y]:yoff_list[y+1],xoff_list[x]:xoff_list[x+1]]
#             xoff = xoff_list[offlist_idx[0]]
#             yoff = ystart+int(i/len(xoff_list))
#             print(yoff_list[yoff], xoff_list[i%len(xoff_list)])
#             tile=data[yoff_list[yoff]:yoff_list[yoff]+BLOCK_SIZE,xoff_list[i%len(xoff_list)]:xoff_list[i%len(xoff_list)]+BLOCK_SIZE]
#         xoff_start=xoff_end
#         yoff_start=yoff_end
#         if n == batch_num-1:
#             last_loc = [(len(xoff_list)-1)-(batch_size*i%(len(xoff_list)-1)),(len(yoff_list)-1)-int(batch_size*i/(len(xoff_list)-1))-1]
#             xoff_start = xoff_list[last_loc[0]]
#             yoff_start = yoff_list[last_loc[1]]
#         print('time:',(datetime.now()-s))
    
    