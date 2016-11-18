paths.dofile('layers/Residual.lua')

-- n是hourglass的深度
-- f means number of features by zf
-- inp means input image
local function hourglass(n, f, inp)
    -- Upper branch
    -- 这个就是在maxpooling之前进行Residual得到的高分辨率的结果 by zf
    local up1 = inp
    -- opt.nModules means Number of residual modules at each location in the hourglass
    -- 重复运用Residual nMoudules次 by zf
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    -- 不断地缩小图像
    -- max pooling一次 by zf
    local low1 = nnlib.SpatialMaxPooling(2,2,2,2)(inp)
    -- 重复运用Residual nMoudules次 by zf
    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else-- 到达最低分辨率的时候重复运用Residual nMoudules次 by zf
        low2 = low1
        -- 重复运用Residual nMoudules次 by zf
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
    end

    -- 降到最低分辨率之后再重复运用Residual nMoudules次 by zf
    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    -- 然后再上采样
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    -- 将高分辨率的feature和低分辨率放大的feature求和
    return nn.CAddTable()({up1,up2})
end

-- linear layer, in fact it's a 1x1 convolution stride 1, no padding
-- and the ReLU layer by zf
local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.SpatialConvolution(numIn,numOut,1,1,1,1,0,0)(inp)
    return nnlib.ReLU(true)(nn.SpatialBatchNormalization(numOut)(l))
end

function createModel()

    local inp = nn.Identity()()

    -- Initial processing of the image
    -- input 3 channels output 64 channels 7x7 convolution with padw and padh =2 and stridew and strideh = 3
    local cnv1_ = nnlib.SpatialConvolution(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = nnlib.ReLU(true)(nn.SpatialBatchNormalization(64)(cnv1_))
    -- Residual 64 channels to 128 channels by zf
    local r1 = Residual(64,128)(cnv1)
    -- 2x2 pooling with stridew and strideh = 2 by zf
    local pool = nnlib.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    -- Residual 128 to 128 channels by zf
    local r4 = Residual(128,128)(pool)
    -- Residual 128 to feature number by zf
    local r5 = Residual(128,opt.nFeats)(r4)

    local out = {}
    local inter = r5

    -- opt.nStack means Number of hourglasses to stack by zf
    for i = 1,opt.nStack do
        local hg = hourglass(4,opt.nFeats,inter)

        -- Residual layers at output resolution
        -- 在一个hourglass之后添加nModules个Residual
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        -- 再用一个线性变换层处理一下
        ll = lin(opt.nFeats,opt.nFeats,ll)

        -- Predicted heatmaps
        -- 最后用一个1x1卷积层变换到关节个数个channel
        local tmpOut = nnlib.SpatialConvolution(opt.nFeats,ref.nOutChannels,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < opt.nStack then
            local ll_ = nnlib.SpatialConvolution(opt.nFeats,opt.nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = nnlib.SpatialConvolution(ref.nOutChannels,opt.nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model

end
