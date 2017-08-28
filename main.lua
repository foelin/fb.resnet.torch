--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

paths.dofile('utility/MyLogger.lua')

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

local trainEpochLogger = MyLogger(paths.concat(opt.save, 'train_epoch.log'))
local valEpochLogger = MyLogger(paths.concat(opt.save, 'val_epoch.log'))

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)
   --local trainTop1, trainTop5, trainLoss = 0,0,0
   trainEpochLogger:setFormats{['1_epoch'] = '%11d'}
   trainEpochLogger:add{
    [ '1_epoch'] = epoch,
    [ '2_loss'] = trainLoss,
    [ '3_top1'] = trainTop1,
    [ '4_top5'] = trainTop5,
   }
   print(string.format('Epoch: [%d][TRAIN SUMMARY] '
      .. 'loss %.3f top1 %.3f top5 %.3f\n',
      epoch, trainLoss, trainTop1, trainTop5))
   print('\n')
   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
   

   -- Run model on validation set
   local valTop1, valTop5 = trainer:test(epoch, valLoader)

   local bestModel = false
   if valTop1 < bestTop1 then
      bestModel = true
      bestTop1 = valTop1
      bestTop5 = valTop5
      print(' * Best model ', valTop1, valTop5)
   end

   valEpochLogger:setFormats{['1_epoch'] = '%11d'}
   valEpochLogger:add{
    [ '1_epoch'] = epoch,
    [ '2_top1'] = valTop1,
    [ '3_top5'] = valTop5,
   }
   print(string.format('Epoch: [%d][Val SUMMARY] top1 %.3f top5 %.3f\n', epoch, valTop1, valTop5))
   
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
