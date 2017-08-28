--[[ MyLogger: a simple class to log symbols during training,
        and automate plot generation

Example:
    logger = optim.MyLogger('somefile.log')    -- file to save stuff

    for i = 1,N do                           -- log some symbols during
        train_error = ...                     -- training/valing
        val_error = ...
        logger:add{['training error'] = train_error,
            ['val error'] = val_error}
    end

    logger:style{['training error'] = '-',   -- define styles for plots
                 ['val error'] = '-'}
    logger:plot()                            -- and plot

---- OR ---

    logger = optim.MyLogger('somefile.log')    -- file to save stuff
    logger:setNames{'training error', 'val error'}

    for i = 1,N do                           -- log some symbols during
       train_error = ...                     -- training/valing
       val_error = ...
       logger:add{train_error, val_error}
    end

    logger:style{'-', '-'}                   -- define styles for plots
    logger:plot()                            -- and plot

-----------

    logger:setlogscale(true)                 -- enable logscale on Y-axis
    logger:plot()                            -- and plot
]]
require 'xlua'
local MyLogger = torch.class('MyLogger')

function MyLogger:__init(filename, timestamp, append)
   if filename then
      self.name = filename
      os.execute('mkdir ' .. (sys.uname() ~= 'windows' and '-p ' or '') .. ' "' .. paths.dirname(filename) .. '"')
      if timestamp then
         -- append timestamp to create unique log file
         filename = filename .. '-'..os.date("%Y_%m_%d_%X")
      end
      if append then
        self.append = true
        self.file = io.open(filename,'a')
      else
        self.file = io.open(filename,'w')
      end
      self.epsfile = self.name .. '.eps'
   else
      self.file = io.stdout
      self.name = 'stdout'
      print('<MyLogger> warning: no path provided, logging to std out')
   end
   self.empty = true
   self.s_key = {}
   self.formats = {}
   self.symbols = {}
   self.styles = {}
   self.names = {}
   self.idx = {}
   self.figure = nil
   self.showPlot = true
   self.plotRawCmd = nil
   self.defaultStyle = '+'
   self.logscale = false
end

function MyLogger:setNames(names)
   self.names = names
   self.empty = false
   self.nsymbols = #names

   --sort the names
    for k in pairs(names) do table.insert(self.s_key, k) end
    table.sort(self.s_key)

   for i,k in ipairs(self.s_key) do
      self.file:write(k .. '\t')
      self.symbols[k] = {}
      self.styles[k] = {self.defaultStyle}
      self.idx[k] = k
   end
   self.file:write('\n')
   self.file:flush()
   return self
end

function MyLogger:setFormats(formats)
   
   for k,v in pairs(formats) do
      self.formats[k] = formats[k]
   end
   return self
end


function MyLogger:add(symbols)
   -- (1) first time ? print symbols' names on first row
   if self.empty then
      self.empty = false
      self.nsymbols = #symbols

      --sort the symbols
      for k in pairs(symbols) do table.insert(self.s_key, k) end
      table.sort(self.s_key)

      if self.append then
        for i,k in ipairs(self.s_key) do
           self.symbols[k] = {}
           self.styles[k] = {self.defaultStyle}
           self.names[k] = k
        end
        self.idx = self.names
      else
         for i,k in ipairs(self.s_key) do
           self.file:write(k .. '\t')
           self.symbols[k] = {}
           self.styles[k] = {self.defaultStyle}
           self.names[k] = k
        end
        self.idx = self.names
        self.file:write('\n')
      end
   end
   -- (2) print all symbols on one row
   for i,k in pairs(self.s_key) do
      local val = symbols[k]
      if self.formats[k] ~= nil then
        self.file:write(string.format(self.formats[k],val) .. '\t')
      else
        if type(val) == 'number' then
          self.file:write(string.format('%11.4e',val) .. '\t')
        elseif type(val) == 'string' then
          self.file:write(val .. '\t')
        else
          xlua.error('can only log numbers and strings', 'MyLogger')
        end
      end
   end
   self.file:write('\n')
   self.file:flush()
   -- (3) save symbols in internal table
   for i,k in pairs(self.s_key) do
      table.insert(self.symbols[k], symbols[k])
   end
end


--TODO add s_key
function MyLogger:style(symbols)
   for name,style in pairs(symbols) do
      if type(style) == 'string' then
         self.styles[name] = {style}
      elseif type(style) == 'table' then
         self.styles[name] = style
      else
         xlua.error('style should be a string or a table of strings','MyLogger')
      end
   end
   return self
end

function MyLogger:setlogscale(state)
   self.logscale = state
end

function MyLogger:display(state)
   self.showPlot = state
end

--TODO add s_key
function MyLogger:plot(...)
   if not xlua.require('gnuplot') then
      if not self.warned then
         print('<MyLogger> warning: cannot plot with this version of Torch')
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
      if self.showPlot then
         self.figure = gnuplot.figure(self.figure)
         if self.logscale then gnuplot.logscale('on') end
         gnuplot.plot(plots)
         if self.plotRawCmd then gnuplot.raw(self.plotRawCmd) end
         gnuplot.grid('on')
         gnuplot.title('<MyLogger::' .. self.name .. '>')
      end
      if self.epsfile then
         os.execute('rm -f "' .. self.epsfile .. '"')
         local epsfig = gnuplot.epsfigure(self.epsfile)
         if self.logscale then gnuplot.logscale('on') end
         gnuplot.plot(plots)
         if self.plotRawCmd then gnuplot.raw(self.plotRawCmd) end
         gnuplot.grid('on')
         gnuplot.title('<MyLogger::' .. self.name .. '>')
         gnuplot.plotflush()
         gnuplot.close(epsfig)
      end
   end
end
