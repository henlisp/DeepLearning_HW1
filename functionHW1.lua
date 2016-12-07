require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

local opt = lapp[[
    -r,--learningRate  (default 0.1)        learning rate
    -m,--momentum      (default 0.9)        momentum
]]


-----### Load the data

local mnist = require 'mnist';
local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

	--normalizing our data
local mean = trainData:mean()
local std = trainData:std()
--trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


------ ### Load a saved model, run it on the test set, and return the average error : 
local filename = "model.t7"
model = torch.load(filename)

batchSize = 128

---- ### Classification criterion

criterion = nn.ClassNLLCriterion():cuda()

function forwardNet(data, labels, train)
    
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0

    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
    end

	confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid

    return avgLoss, avgError, tostring(confusion)
end


	
--- ### return the average error on the test set

	testLoss, testError, confusion = forwardNet(testData, testLabels, false)
	print('Test error: ' .. testError, 'Test Loss: ' .. testLoss)
