-- 递归调用自己，如果tabel内部嵌入tabel的话最后才调用fn
function applyFn(fn, t, t2)
    -- Apply an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------
-- 生成平移矩阵
function getTransform(center, scale, rot, res)
    local h = 200 * scale

    -- 平移和缩放矩阵
    local t = torch.eye(3)

    -- Scaling
    -- 对角线是缩放
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    -- 平移是在第三列
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    -- 设置旋转
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        -- 将角度转换为rad
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        -- t_是平移一下
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        -- t_inv是逆平移
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        -- 先平移，然后再确保旋转的中心在中心位置，然后在旋转，旋转完毕后在平移回去
        t = t_inv * r * t_ * t
    end

    return t
end

-- 对坐标点进行平移、旋转和缩放变换
function transform(pt, center, scale, rot, res, invert)
    -- 3*1的向量
    local pt_ = torch.ones(3)
    -- x和y赋值到pt_并且减去1
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1
    -- 生成平移矩阵
    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    -- 给变换过之后的坐标加上一个很小的值
    local new_point = (t*pt_):sub(1,2):add(1e-4)
    -- 变换后的坐标的每个值加上1，恢复回去
    return new_point:int():add(1)
end

function transformPreds(coords, center, scale, res)
    -- 获取坐标的原始形状
    local origDims = coords:size()
    -- 将坐标转换为n行，2列的形式
    coords = coords:view(-1,2)
    -- 复制到newCoords中
    local newCoords = coords:clone()
    -- 遍历每一个坐标
    for i = 1,coords:size(1) do
        -- 对每一个坐标点进行变换一下
        newCoords[i] = transform(coords[i], center, scale, 0, res, 1)
    end
    -- 将形状设置为原来的形状
    return newCoords:view(origDims)
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function checkDims(dims)
    return dims[3] < dims[4] and dims[5] < dims[6]
end

function crop(img, center, scale, rot, res)
    -- 获取图像的维度
    local ndim = img:nDimension()
    -- 如果是2维，则是黑白图像
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    -- 获取高度和宽度  size(2)是高度，size(3)是宽度
    local ht,wd = img:size(2), img:size(3)
    local tmpImg = img
    -- 缩放尺度，为啥是200乘以尺度？
    local scaleFactor = (200 * scale) / res
    if scaleFactor < 2 then scaleFactor = 1
    else tmpImg = image.scale(img,math.ceil(math.max(ht,wd) / scaleFactor)) end

    ht,wd = tmpImg:size(2),tmpImg:size(3)
    -- 经过缩放之后的center和scale
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    -- 对{1,1}和{res+1,res+1}进行变换
    -- ul是左上角，br是右下角
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)

    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then
        ul = ul - pad
        br = br + pad
    end

    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- Check that dimensions are okay
    if not (checkDims(new_) and checkDims(old_)) then
        return torch.zeros(img:size(1),res,res)
    end
    local newImg = torch.zeros(img:size(1), br[2] - ul[2] + 1, br[1] - ul[1] + 1)
    if rot == 0 and scaleFactor > 2 then newImg = torch.zeros(img:size(1),res,res) end
    newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_)))

    if rot ~= 0 then
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad,newImg:size(2)-pad,pad,newImg:size(3)-pad)
    end

    newImg = image.scale(newImg,res,res)
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
    return newImg
end

function crop2(img, center, scale, rot, res)
    local ul = transform({1,1}, center, scale, 0, res, true)
    local br = transform({res+1,res+1}, center, scale, 0, res, true)


    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then
        ul = ul - pad
        br = br + pad
    end

    local newDim,newImg,ht,wd

    if img:size():size() > 2 then
        newDim = torch.IntTensor({img:size(1), br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2],newDim[3])
        ht = img:size(2)
        wd = img:size(3)
    else
        newDim = torch.IntTensor({br[2] - ul[2], br[1] - ul[1]})
        newImg = torch.zeros(newDim[1],newDim[2])
        ht = img:size(1)
        wd = img:size(2)
    end

    local newX = torch.Tensor({math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]})
    local newY = torch.Tensor({math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2]})
    local oldX = torch.Tensor({math.max(1, ul[1]), math.min(br[1], wd+1) - 1})
    local oldY = torch.Tensor({math.max(1, ul[2]), math.min(br[2], ht+1) - 1})

    if newDim:size(1) > 2 then
        newImg:sub(1,newDim[1],newY[1],newY[2],newX[1],newX[2]):copy(img:sub(1,newDim[1],oldY[1],oldY[2],oldX[1],oldX[2]))
    else
        newImg:sub(newY[1],newY[2],newX[1],newX[2]):copy(img:sub(oldY[1],oldY[2],oldX[1],oldX[2]))
    end

    if rot ~= 0 then
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        if newDim:size(1) > 2 then
            newImg = newImg:sub(1,newDim[1],pad,newDim[2]-pad,pad,newDim[3]-pad)
        else
            newImg = newImg:sub(pad,newDim[1]-pad,pad,newDim[2]-pad)
        end
    end

    newImg = image.scale(newImg,res,res)
    return newImg
end

function twoPointCrop(img, s, pt1, pt2, pad, res)
    local center = (pt1 + pt2) / 2
    local scale = math.max(20*s,torch.norm(pt1 - pt2)) * .007
    scale = scale * pad
    local angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)
end

-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------
-- 利用maxpooling来进行nms
-- 首先需要给heatmap加一维才能使用内置的maxpoling来处理
-- hm是所有的heatmap
-- hmIdx是某个channel的heatmap
-- c是表示是否转换坐标
function localMaxes(hm, n, c, s, hmIdx, nmsWindowSize)
    -- Set up max network for NMS
    local nmsWindowSize = nmsWindowSize or 3
    local nmsPad = (nmsWindowSize - 1)/2
    local maxlayer = nn.Sequential()
    if cudnn then
        maxlayer:add(cudnn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad, nmsPad))
        maxlayer:cuda()
    else
        maxlayer:add(nn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad,nmsPad))
        maxlayer:float()
    end
    maxlayer:evaluate()

    -- 将hm的形状转换为table
    local hmSize = torch.totable(hm:size())
    -- unpack是去掉维度数值table的索引值
    -- 实际上本句是为了初始化新的hm，而新的hm是多了一维的，只是为了后面能够调用SpatialMaxPooling
    hm = torch.Tensor(1,unpack(hmSize)):copy(hm):float()
    -- 如果指定了hmIdx的话，则取出hmIdx所指定的那个channel的heatmap取出来放到hm
    if hmIdx then hm = hm:sub(1,-1,hmIdx,hmIdx) end
    local hmDim = hm:size()
    local max_out
    -- First do nms
    -- 调用maxpooling进行nms
    if cudnn then
        max_out = maxlayer:forward(hm:cuda())
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end
    -- torch.eq(hm, max_out:float())是判断hm中的每一个元素是否为最大值，如果是则为1，否则为0，形成一个与hm一样大小的矩阵
    -- torch.cmul是将hm和torch.eq的结果矩阵点乘
    -- 只保留了最大值
    -- 因为hm多加了一维，这里最后用了[1]去掉了当初的第一维
    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    -- hmDim[2]是channel的个数也就是关节的个数
    local predCoords = torch.Tensor(hmDim[2], n, 2)
    local predScores = torch.Tensor(hmDim[2], n)
    -- 遍历每一个channel的heatmap
    for i = 1, hmDim[2] do
        -- 首先将heatmap变成一个向量
        local nms_flat = nms[i]:view(nms[i]:nElement())
        -- 在第1维进行降序(false为升序,true为降序)排列
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            -- x=(idxs[j]-1) % hmSize[3] + 1
            -- y=math.floor((idxs[j]-1) / hmSize[3]) + 1 
            local pt = {(idxs[j]-1) % hmSize[3] + 1, math.floor((idxs[j]-1) / hmSize[3]) + 1 }
            if c then
                predCoords[i][j] = transform(pt, c, s, 0, hmSize[#hmSize], true)
            else
                predCoords[i][j] = torch.Tensor(pt)
            end
            predScores[i][j] = vals[j]
        end
    end
    return predCoords, predScores
end

-------------------------------------------------------------------------------
-- Draw gaussian
-------------------------------------------------------------------------------
-- 给图像以指定的点为中心加上高斯噪声
function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    -- 3Sigma水平代表了99.73%的概率
    -- 左上坐标
    local ul = {math.floor(pt[1] - 3 * sigma), math.floor(pt[2] - 3 * sigma)}
    -- 右下坐标
    local br = {math.floor(pt[1] + 3 * sigma), math.floor(pt[2] + 3 * sigma)}
    -- If not, return the image as is
    -- 理论上来说左上x应该小于图像的x，y应该小于图像的y
    -- 右下的x应该大于1并且右下的y应该大于1
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    -- 因为是3sigma，所以大小是6sigma+1,+1是因为有中心点
    local size = 6 * sigma + 1
    local g = image.gaussian(size) -- , 1 / size, 1)
    -- Usable gaussian range
    -- 可用的高斯范围x的起点和终点
    -- 这里为啥这么做？
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    -- 在图像中的范围
    -- 用左上和右下来指定的
    -- 左上的x和右下的x来确定x的开始和结束
    -- 左上的y和右下的y来确定y的开始和结束
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    -- 图像的高斯部分加上高斯噪声
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):add(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    -- 所有值大于1的都设置为1
    img[img:gt(1)] = 1
    return img
end

function drawLine(img, pt1, pt2, width, color)
    -- 如果是灰度图则设置则将图像设置成和彩色图一样的结构即nchannels,height,width
    -- 便于后面统一处理
    if img:nDimension() == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local nChannels = img:size(1)
    color = color or torch.ones(nChannels)

    -- 将点坐标转换为tensor
    if type(pt1) == 'table' then pt1 = torch.Tensor(pt1) end
    if type(pt2) == 'table' then pt2 = torch.Tensor(pt2) end

    -- 求点1到点2的方向向量
    m = pt1:dist(pt2)
    dy = (pt2[2] - pt1[2])/m
    dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            y_idx = torch.ceil(start_pt1[2]+dy*i)
            x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 
            and y_idx < img:size(2) and x_idx < img:size(3) then
                for j = 1,nChannels do img[j]:sub(y_idx-1,y_idx,x_idx-1,x_idx):fill(color[j]) end
            end
        end
    end
end

function drawSkeleton(img, preds, scores)
    preds = preds:clone():add(1) -- Account for 1-indexing in lua
    local pairRef = dataset.skeletonRef
    local actThresh = 0.05
    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if scores[pairRef[i][1]] > actThresh and scores[pairRef[i][2]] > actThresh then
            -- Set appropriate line color
            local color
            if pairRef[i][3] == 1 then color = {0,.3,1}
            elseif pairRef[i][3] == 2 then color = {1,.3,0}
            elseif pairRef[i][3] == 3 then color = {0,0,1}
            elseif pairRef[i][3] == 4 then color = {1,0,0}
            else color = {.7,0,.7} end
            -- Draw line
            drawLine(img, preds[pairRef[i][1]], preds[pairRef[i][2]], 4, color, 0)
        end
    end
    return img
end

-- 显示heatmap
function heatmapVisualization(set, idx, pred, inp, gt)
    local set = set or 'valid'
    local hmImg
    local tmpInp,tmpHm
    if not inp then
        inp, gt = loadData(set,{idx})
        inp = inp[1]
        gt = gt[1][1]
        tmpInp,tmpHm = inp,gt
    else
        tmpInp = inp
        tmpHm = gt or pred
    end
    -- nOut是关节的个数，res是heatmap的宽度或者高度，因为宽高是一样大
    local nOut,res = tmpHm:size(1),tmpHm:size(3)
    -- Repeat input image, and darken it to overlay heatmaps
    tmpInp = image.scale(tmpInp,res):mul(.3)
    tmpInp[1][1][1] = 1
    hmImg = tmpInp:repeatTensor(nOut,1,1,1)
    if gt then -- Copy ground truth heatmaps to red channel
        -- 1是red channel
        hmImg:sub(1,-1,1,1):add(gt:clone():mul(.7))
    end
    if pred then -- Copy predicted heatmaps to blue channel
        -- 3是blue channel
        hmImg:sub(1,-1,3,3):add(pred:clone():mul(.7))
    end
    -- Rescale so it is a little easier to see
    hmImg = image.scale(hmImg:view(nOut*3,res,res),256):view(nOut,3,256,256)
    return hmImg, inp
end

-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------
-- 坐标的水平翻转
function shuffleLR(x)
    local dim
    local matchedParts = dataset.flipRef
    if x:nDimension() == 4 or x:nDimension() == 2 then
        dim = 2
    else
        assert(x:nDimension() == 3 or x:nDimension() == 1)
        dim = 1
    end

    for i = 1,#matchedParts do
        local idx1, idx2 = unpack(matchedParts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end
    return x
end
-- 图像的水平翻转
function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end
