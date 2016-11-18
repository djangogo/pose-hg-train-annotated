-- Prepare tensors for saving network output
-- 计算测试的样本个数
local validSamples = opt.validIters * opt.validBatch
-- 初始化测试的保存结果的形状
saved = {idxs = torch.Tensor(validSamples),
         preds = torch.Tensor(validSamples, unpack(ref.predDim))}
-- 如果需要保存输入和输出的heatmap则初始化其形状
if opt.saveInput then saved.input = torch.Tensor(validSamples, unpack(ref.inputDim)) end
if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(validSamples, unpack(ref.outputDim[1])) end

-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local output, err, idx
    -- This function returns two tensors.  by zf
    -- One for the flattened learnable parameters flatParameters and another 
    -- for the gradients of the energy wrt to the learnable parameters flatGradParameters.
    local param, gradparam = model:getParameters()
    -- 在model中定义了criterion
    local function evalFn(x) return criterion.output, gradparam end

    if tag == 'train' then
    	-- 设置模块的运行模式为训练模式 by zf
    	-- 这个在使用drop out或者BN的时候需要设置
    	-- This sets the mode of the Module (or sub-modules) to train=true. 
    	-- This is useful for modules like Dropout or BatchNormalization 
    	-- that have a different behaviour during training vs evaluation. by zf
        model:training()
        -- set表示运行模式
        set = 'train'
    else
        model:evaluate()
        if tag == 'predict' then
        	-- 如果是预测结果则初始化saved变量
        	-- 该变量保存预测的结果以及输入的图像和输出的heatmap
            print("==> Generating predictions...")
            local nSamples = dataset:size('test')
            saved = {idxs = torch.Tensor(nSamples),
                     preds = torch.Tensor(nSamples, unpack(ref.predDim))}
            if opt.saveInput then saved.input = torch.Tensor(nSamples, unpack(ref.inputDim)) end
            if opt.saveHeatmaps then saved.heatmaps = torch.Tensor(nSamples, unpack(ref.outputDim[1])) end
            set = 'test'
        else
            set = 'valid'
        end
    end

    -- 获取需要迭代的次数trainIters还是validIters
    -- 在opts.lua中定义
    local nIters = opt[set .. 'Iters']
    -- 每次读取一minibatch数据
    for i,sample in loader[set]:run() do
    	-- 显示进度条
        xlua.progress(i, nIters)
        -- 解包样本
        -- input是输入的图像
        -- label是heatmap
        -- indices是索引
        local input, label, indices = unpack(sample)

        -- 将数据传输到gpu
        if opt.GPU ~= -1 then
            -- Convert to CUDA
            input = applyFn(function (x) return x:cuda() end, input)
            label = applyFn(function (x) return x:cuda() end, label)
        end

        -- Do a forward pass and calculate loss
        local output = model:forward(input)
        local err = criterion:forward(output, label)
        -- 计算平均误差
        avgLoss = avgLoss + err / nIters

        if tag == 'train' then
            -- Training: Do backpropagation and optimization
            -- 清除梯度
            model:zeroGradParameters()
            -- 反传一次
            model:backward(input, criterion:backward(output, label))
            -- 更新参数
            optfn(evalFn, param, optimState)
        else
            -- Validation: Get flipped output
            -- 将output重新复制一份
            output = applyFn(function (x) return x:clone() end, output)
            -- 将输入图像flip然后前传得到输出
            local flippedOut = model:forward(flip(input))
            -- 将输出的heatmap通过函数shuffleLR一下
            flippedOut = applyFn(function (x) return flip(shuffleLR(x)) end, flippedOut)
            -- 将output和flippedOut相加，然后除以2
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

            -- Save sample
            -- 获取validBatch的大小
            -- Mini-batch size for validation by zf
            -- bs是batchsize
            local bs = opt[set .. 'Batch']
            -- 数据开始的索引为tmpIdx
            local tmpIdx = (i-1) * bs + 1
            local tmpOut = output
            if type(tmpOut) == 'table' then tmpOut = output[#output] end
            -- 将输入进行保存，保存到saved.input
            if opt.saveInput then saved.input:sub(tmpIdx, tmpIdx+bs-1):copy(input) end
            -- 将输出进行保存，保存到saved.heatmap
            if opt.saveHeatmaps then saved.heatmaps:sub(tmpIdx, tmpIdx+bs-1):copy(tmpOut) end
            -- 保存索引
            saved.idxs:sub(tmpIdx, tmpIdx+bs-1):copy(indices)
            -- 保存预测结果
            saved.preds:sub(tmpIdx, tmpIdx+bs-1):copy(postprocess(set,indices,output))
        end

        -- Calculate accuracy
        avgAcc = avgAcc + accuracy(output, label) / nIters
    end


    -- Print and log some useful metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if ref.log[set] then
        table.insert(opt.acc[set], avgAcc)
        ref.log[set]:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    -- 如果是测试的话，则保存优化的参数，选项以及模型，然后把预测结果也保存
    -- 保存的名字是model_epoch.t7
    -- 预测结果是final_pred.h5
    if (tag == 'valid' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0) or tag == 'predict' then
        -- Take a snapshot
        model:clearState()
        torch.save(paths.concat(opt.save, 'options.t7'), opt)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
        local predFilename = 'preds.h5'
        if tag == 'predict' then predFilename = 'final_' .. predFilename end
        local predFile = hdf5.open(paths.concat(opt.save,predFilename),'w')
        for k,v in pairs(saved) do predFile:write(k,v) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
