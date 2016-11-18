--[[ Logger: a simple class to log symbols during training,
        and automate plot generation

#### Slightly modified from optim.Logger to allow appending to logs ####

Example:
    logger = Logger('somefile.log')    -- file to save stuff

    for i = 1,N do                           -- log some symbols during
        train_error = ...                     -- training/testing
        test_error = ...
        logger:add{['training error'] = train_error,
            ['test error'] = test_error}
    end

    logger:style{['training error'] = '-',   -- define styles for plots
                 ['test error'] = '-'}
    logger:plot()                            -- and plot

---- OR ---

    logger = optim.Logger('somefile.log')    -- file to save stuff
    logger:setNames{'training error', 'test error'}

    for i = 1,N do                           -- log some symbols during
       train_error = ...                     -- training/testing
       test_error = ...
       logger:add{train_error, test_error}
    end

    logger:style{'-', '-'}                   -- define styles for plots
    logger:plot()                            -- and plot
]]
require 'xlua'
local Logger = torch.class('Logger')

-- 初始化，设置文件名称以及是否添加时间戳，此外还设置是否在之前的接触上继续添加日志
function Logger:__init(filename, continue, timestamp)
  -- 如果指定了文件名则创建该目录
   if filename then
      self.name = filename
      os.execute('mkdir -p "' .. paths.dirname(filename) .. '"')
      -- 如果指定了时间戳则在文件名后加时间戳
      if timestamp then
         -- append timestamp to create unique log file
         filename = filename .. '-'..os.date("%Y_%m_%d_%X")
      end
      -- 是不是在前面的日志基础之上继续添加日志
      if continue then
      	self.file = io.open(filename,'a')
      else
      	self.file = io.open(filename,'w')
      end
      -- 图像文件
      self.epsfile = self.name .. '.eps'
   else
      -- 如果没有指定就用标准输出
      self.file = io.stdout
      self.name = 'stdout'
      print('<Logger> warning: no path provided, logging to std out')
   end
   self.continue = continue
   self.empty = true
   self.symbols = {}
   self.styles = {}
   self.names = {}
   self.idx = {}
   self.figure = nil
end

-- 设置符号名字
function Logger:setNames(names)
  -- 设置给内部变量
   self.names = names
   -- name是否为空
   self.empty = false
   -- 符号个数
   self.nsymbols = #names
   for k,key in pairs(names) do
      -- 显示出names中的每一个
      self.file:write(key .. '\t')
      -- 符号设置为一个空表
      self.symbols[k] = {}
      -- 设置风格为+
      self.styles[k] = {'+'}
      -- 把索引加进去idx
      self.idx[key] = k
   end
   self.file:write('\n')
   self.file:flush()
end

-- 加入要记录的符号
function Logger:add(symbols)
   -- (1) first time ? print symbols' names on first row
   if self.empty then
      self.empty = false
      self.nsymbols = #symbols
      -- 遍历每一个符号
      for k,val in pairs(symbols) do
        -- 如果continue是false则显示symbol的索引
	       if not self.continue then self.file:write(k .. '\t') end
         self.symbols[k] = {}
         self.styles[k] = {'+'}
         self.names[k] = k
      end
      self.idx = self.names
      if not self.continue then self.file:write('\n') end
   end
   -- (2) print all symbols on one row
   -- 显示需要记录的变量符号
   for k,val in pairs(symbols) do
      if type(val) == 'number' then
         self.file:write(string.format('%11.4e',val) .. '\t')
      elseif type(val) == 'string' then
         self.file:write(val .. '\t')
      else
         xlua.error('can only log numbers and strings', 'Logger')
      end
   end
   self.file:write('\n')
   self.file:flush()
   -- (3) save symbols in internal table
   for k,val in pairs(symbols) do
      table.insert(self.symbols[k], val)
   end
end

function Logger:style(symbols)
   for name,style in pairs(symbols) do
      if type(style) == 'string' then
         self.styles[name] = {style}
      elseif type(style) == 'table' then
         self.styles[name] = style
      else
         xlua.error('style should be a string or a table of strings','Logger')
      end
   end
end

function Logger:plot(...)
   if not xlua.require('gnuplot') then
      if not self.warned then
         print('<Logger> warning: cannot plot with this version of Torch')
         self.warned = true
      end
      return
   end
   local plotit = false
   local plots = {}
   local plotsymbol =
      function(name,list)
         if #list > 1 then
            local nelts = #list
            local plot_y = torch.Tensor(nelts)
            for i = 1,nelts do
               plot_y[i] = list[i]
            end
            for _,style in ipairs(self.styles[name]) do
               table.insert(plots, {self.names[name], plot_y, style})
            end
            plotit = true
         end
      end
   local args = {...}
   if not args[1] then -- plot all symbols
      for name,list in pairs(self.symbols) do
         plotsymbol(name,list)
      end
   else -- plot given symbols
      for _,name in ipairs(args) do
         plotsymbol(self.idx[name], self.symbols[self.idx[name]])
      end
   end
   if plotit then
      self.figure = gnuplot.figure(self.figure)
      gnuplot.plot(plots)
      gnuplot.grid('on')
      gnuplot.title('<Logger::' .. self.name .. '>')
      if self.epsfile then
         os.execute('rm -f "' .. self.epsfile .. '"')
         local epsfig = gnuplot.epsfigure(self.epsfile)
         gnuplot.plot(plots)
         gnuplot.grid('on')
         gnuplot.title('<Logger::' .. self.name .. '>')
         gnuplot.plotflush()
         gnuplot.close(epsfig)
      end
   end
end
