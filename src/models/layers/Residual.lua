local conv = nnlib.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        -- 1x1 convolution byzf
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        -- 3x3 convolution by zf
        -- padw and padh = 1
        -- dw and dh = 1
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        -- 1x1 convolution by zf
        :add(conv(numOut/2,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
        -- 1x1 convolution by zf
            :add(conv(numIn,numOut,1,1))
    end
end

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

