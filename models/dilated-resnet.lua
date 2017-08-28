--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local DilatedConvolution = nn.SpatialDilatedConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end

   local function dilated_basicblock(n, scale, d_type)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()

      if d_type == 4 then
         --no shortcut
         s:add(DilatedConvolution(nInputPlane,n,3,3,1,1,scale,scale,scale,scale))
            s:add(SBatchNorm(n))
            s:add(ReLU(true))
            s:add(DilatedConvolution(n,n,3,3,1,1,scale,scale,scale,scale))
            s:add(SBatchNorm(n))
            s:add(ReLU(true)) -- note that we need relu
         return s

      else

         if d_type == 1 then
            s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
            s:add(SBatchNorm(n))
            s:add(ReLU(true))
            s:add(DilatedConvolution(n,n,3,3,1,1,scale,scale,scale,scale))
            s:add(SBatchNorm(n))
         elseif d_type == 2 then
            s:add(DilatedConvolution(nInputPlane,n,3,3,1,1,scale/2,scale/2,scale/2,scale/2))
            s:add(SBatchNorm(n))
            s:add(ReLU(true))
            s:add(DilatedConvolution(n,n,3,3,1,1,scale,scale,scale,scale))
            s:add(SBatchNorm(n))
         elseif d_type == 3 then
            s:add(DilatedConvolution(nInputPlane,n,3,3,1,1,scale,scale,scale,scale))
            s:add(SBatchNorm(n))
            s:add(ReLU(true))
            s:add(DilatedConvolution(n,n,3,3,1,1,scale,scale,scale,scale))
            s:add(SBatchNorm(n))
         
         end

         return nn.Sequential()
            :add(nn.ConcatTable()
               :add(s)
               :add(shortcut(nInputPlane, n, 1)))
            :add(nn.CAddTable(true))
            :add(ReLU(true))

      end     
   end


   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()
      s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n*4,1,1,1,1,0,0))
      s:add(SBatchNorm(n * 4))

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end


   local function dilated_bottleneck(n, scale, d_type)
      local nInputPlane = iChannels
      iChannels = n * 4

      local s = nn.Sequential()

      if d_type == 1 then
         s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
         s:add(SBatchNorm(n))
         s:add(ReLU(true))
         s:add(Convolution(n,n,3,3,1,1,1,1))
         s:add(SBatchNorm(n))
         s:add(ReLU(true))
         s:add(Convolution(n,n*4,1,1,1,1,0,0))
         s:add(SBatchNorm(n * 4))
      elseif d_type == 2 then
         s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
         s:add(SBatchNorm(n))
         s:add(ReLU(true))
         s:add(DilatedConvolution(n,n,3,3,1,1,scale/2,scale/2,scale/2,scale/2))
         s:add(SBatchNorm(n))
         s:add(ReLU(true))
         s:add(Convolution(n,n*4,1,1,1,1,0,0))
         s:add(SBatchNorm(n * 4))
      elseif d_type == 3 then
         s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
         s:add(SBatchNorm(n))
         s:add(ReLU(true))
         s:add(DilatedConvolution(n,n,3,3,1,1,scale,scale,scale,scale))
         s:add(SBatchNorm(n))
         s:add(ReLU(true))
         s:add(Convolution(n,n*4,1,1,1,1,0,0))
         s:add(SBatchNorm(n * 4))
      end 

      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n * 4, 1)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
   end




   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local function dilated_layer(block, features, count, scale, first_dilated_layer)
      local s = nn.Sequential()
      
      --first block
      if first_dilated_layer then
         s:add(block(features, scale, 1))
      else
         s:add(block(features, scale, 2))
      end

      --other block
      for i=2,count do
         s:add(block(features, scale, 3))
      end
      return s
   end

   local function end_dilated_layer(block, features, count, scale, with_residual)
      local s = nn.Sequential()
      
      if with_residual then
         for i=1,count do
            s:add(block(features, scale, 3))
         end
      else
         for i=1,count do
            s:add(block(features, scale, 4))
         end
      end      
      return s
   end


   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      -- Configurations for ResNet:
      --  num. residual blocks, num features, residual block function
      local cfg = {
         [18]  = {{2, 2, 2, 2}, 512,   {basicblock, basicblock, dilated_basicblock, dilated_basicblock}},
         [34]  = {{3, 4, 6, 3}, 512,   {basicblock, basicblock, dilated_basicblock, dilated_basicblock}},
         [50]  = {{3, 4, 6, 3}, 2048,  {bottleneck, bottleneck, dilated_bottleneck, dilated_bottleneck}},
         [101] = {{3, 4, 23, 3}, 2048, {bottleneck, bottleneck, dilated_bottleneck, dilated_bottleneck}},
         [152] = {{3, 8, 36, 3}, 2048, {bottleneck, bottleneck, dilated_bottleneck, dilated_bottleneck}},
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local def, nFeatures, block = table.unpack(cfg[depth])
      
      print(' | ResNet-' .. depth .. ' ImageNet')

      -- The ResNet ImageNet model
      
      if opt.netType == 'A' then
         iChannels = 64

         model:add(Convolution(3,64,7,7,2,2,3,3))
         model:add(SBatchNorm(64))
         model:add(ReLU(true))
         model:add(Max(3,3,2,2,1,1))
         model:add(layer(block[1], 64, def[1]))
         model:add(layer(block[2], 128, def[2], 2))
         model:add(dilated_layer(block[3], 256, def[3], 2, true))
         model:add(dilated_layer(block[4], 512, def[4], 4, false))
         model:add(Avg(28, 28, 1, 1))
         model:add(nn.View(nFeatures):setNumInputDims(3))
         model:add(nn.Linear(nFeatures, 1000))
      elseif opt.netType == 'B' then
         iChannels = 16

         model:add(Convolution(3,16,7,7,1,1,3,3))
         model:add(SBatchNorm(16))
         model:add(ReLU(true))
         model:add(layer(basicblock,16,1))
         model:add(layer(basicblock,32,1,2))
         model:add(layer(block[1], 64, def[1], 2))
         model:add(layer(block[2], 128, def[2], 2))
         model:add(dilated_layer(block[3], 256, def[3], 2, true))
         model:add(dilated_layer(block[4], 512, def[4], 4, false))
         model:add(end_dilated_layer(dilated_basicblock, 512, 1, 2, true))
         model:add(end_dilated_layer(dilated_basicblock, 512, 1, 1, true))
         model:add(Avg(28, 28, 1, 1))
         model:add(nn.View(nFeatures):setNumInputDims(3))
         model:add(nn.Linear(nFeatures, 1000))
      elseif opt.netType == 'C' then
         iChannels = 16

         model:add(Convolution(3,16,7,7,1,1,3,3))
         model:add(SBatchNorm(16))
         model:add(ReLU(true))
         model:add(layer(basicblock,16,1))
         model:add(layer(basicblock,32,1,2))
         model:add(layer(block[1], 64, def[1], 2))
         model:add(layer(block[2], 128, def[2], 2))
         model:add(dilated_layer(block[3], 256, def[3], 2, true))
         model:add(dilated_layer(block[4], 512, def[4], 4, false))
         model:add(end_dilated_layer(dilated_basicblock, 512, 1, 2, false))
         model:add(end_dilated_layer(dilated_basicblock, 512, 1, 1, false))
         model:add(Avg(28, 28, 1, 1))
         model:add(nn.View(nFeatures):setNumInputDims(3))
         model:add(nn.Linear(nFeatures, 1000))
      else
         error('Undefined netType: ' .. opt.netType)
      end


   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(dilated_layer(dilated_basicblock, 32, n, 2, true))
      model:add(dilated_layer(dilated_basicblock, 64, n, 4, false))
      model:add(Avg(32, 32, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-100')

      -- The ResNet CIFAR-100 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(SBatchNorm(16))
      model:add(ReLU(true))
      model:add(layer(basicblock, 16, n))
      model:add(dilated_layer(dilated_basicblock, 32, n, 2, true))
      model:add(dilated_layer(dilated_basicblock, 64, n, 4, false))
      model:add(Avg(32, 32, 1, 1))
      model:add(nn.View(64):setNumInputDims(3))
      model:add(nn.Linear(64, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   ConvInit('cudnn.SpatialDilatedConvolution')
   ConvInit('nn.SpatialDilatedConvolution')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
