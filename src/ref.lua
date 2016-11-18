require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'
ffi = require 'ffi'

-- 设置默认数据类型为float
torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
projectDir = paths.concat(os.getenv('HOME'),'pose-hg-train')

-- Process command line arguments, load helper functions
-- 设置选项
paths.dofile('opts.lua')
-- 对图像进行预处理的
paths.dofile('util/img.lua')
-- 显示pck曲线的
paths.dofile('util/eval.lua')
if not Logger then paths.dofile('util/Logger.lua') end

-- Random number seed
if opt.manualSeed ~= -1 then torch.manualSeed(opt.manualSeed)
else torch.seed() end                           

-- Initialize dataset
if not dataset then
    local Dataset = paths.dofile(projectDir .. '/src/util/dataset/' .. opt.dataset .. '.lua')
    dataset = Dataset()
end

-- Global reference (may be updated in the task file below)
-- 全局变量，设置输出的channels以及网络的输入维度和输出维度
if not ref then
    ref = {}
    ref.nOutChannels = dataset.nJoints
    ref.inputDim = {3, opt.inputRes, opt.inputRes}
    ref.outputDim = {ref.nOutChannels, opt.outputRes, opt.outputRes}
end

-- Load up task specific variables / functions
paths.dofile('util/' .. opt.task .. '.lua')

-- Optimization function and hyperparameters
-- 使用什么优化算法以及优化的参数
optfn = optim[opt.optMethod]
if not optimState then
    optimState = {
        learningRate = opt.LR,
        learningRateDecay = opt.LRdecay,
        momentum = opt.momentum,
        weightDecay = opt.weightDecay,
        alpha = opt.alpha,
        epsilon = opt.epsilon
    }
end

-- Print out input / output tensor sizes
-- 打印输入和输出张量的大小
if not ref.alreadyChecked then
    local function printDims(prefix,d)
        -- Helper for printing out tensor dimensions
        if type(d[1]) == "table" then
            print(prefix .. "table")
            for i = 1,#d do
                printDims("\t Entry " .. i .. " is a ", d[i])
            end
        else
            local s = ""
            if #d == 0 then s = "single value"
            elseif #d == 1 then s = string.format("vector of length: %d", d[1])
            else
                s = string.format("tensor with dimensions: %d", d[1])
                for i = 2,table.getn(d) do s = s .. string.format(" x %d", d[i]) end
            end
            print(prefix .. s)
        end
    end

    printDims("Input is a ", ref.inputDim)
    printDims("Output is a ", ref.outputDim)
    ref.alreadyChecked = true
end
