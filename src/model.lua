--- Load up network model or initialize from scratch
-- 将hourglass模型加载进来
paths.dofile('models/' .. opt.netType .. '.lua')

-- Continuing an experiment where it left off
-- 是否加载之前停下来的模型，继续训练
if opt.continue or opt.branch ~= 'none' then
    --local prevModel = opt.load .. '/final_model.t7'
    local prevModel = opt.load .. '/model_40.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)

-- Or a path to previously trained model is provided
-- 是否加载之前训练好的模型
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

-- Or we're starting fresh
else -- 直接从头开始训练
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
end

-- Criterion (can be set in the opt.task file as well)
-- 损失函数
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
    
    cudnn.fastest = true
    cudnn.benchmark = true
end
