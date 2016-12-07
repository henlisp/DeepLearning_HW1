	local mnist = require 'mnist';
require 'optim'

local opt = lapp[[
    -s,--save          (default 0)          save the model
    -l,--load          (default 0)          load the model
    -o,--optimization  (default "sgd")      optimization type
    -n,--network       (default "n2")       reload pretrained network
    -e,--epoch         (default 20)         number of epochs
    -b,--batch         (default 128)        number of batches
    -r,--learningRate  (default 0.1)        learning rate
    -m,--momentum      (default 0.9)        momentum
]]


local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


----- ### Shuffling data

function shuffle(data, labels) --shuffle data function
    local randomIndexes = torch.randperm(data:size(1)):long() 
    return data:index(1,randomIndexes), labels:index(1,randomIndexes)
end

------  ### Define model and criterion

require 'nn'
require 'cunn'
require 'cudnn'

local inputSize = 28*28
local outputSize = 10


    
------ ### Function that uses a saved model, run it on the test set, and return the average error : 

local filename = "model.t7"
if opt.load == 1 then
    model = torch.load(filename)

batchSize = 128

---- ### Classification criterion

criterion = nn.ClassNLLCriterion():cuda()
	
function forwardNet(data, labels)
    
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    end
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
	
	testLoss, testError, confusion = forwardNet(testData, testLabels)
	print('Test error: ' .. testError, 'Test Loss: ' .. testLoss)

else 	---end




model = nn.Sequential()

if opt.network == 'n1' then
    model:add(nn.View(28 * 28)) --reshapes the image into a vector without copy
    model:add(nn.Linear(inputSize, 64))
    model:add(nn.ReLU())
    model:add(nn.Linear(64, 64))
    model:add(nn.ReLU())

    model:add(nn.Linear(64, 64))
    model:add(nn.ReLU())
    model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
    model:add(nn.Linear(64, outputSize))

elseif opt.network == 'n2' then

    model:add(nn.View(28 * 28))                 -- Reshapes the image into a vector without copy
    local layerSize = {inputSize, 64,128,128}     -- Changed last parameter from 256 to 128
    for i=1, #layerSize-2 do                     -- Substituted #layerSize-1 to #layerSize-2
        model:add(nn.Linear(layerSize[i], layerSize[i+1]))
        model:add(nn.ReLU())
	end
    ---model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
	model:add(nn.Linear(layerSize[#layerSize], outputSize))


--[[ Convolutional Network model ( doesn't work )
elseif opt.network == 'n3' then
    model:add(cudnn.SpatialConvolution(1, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
    model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
    model:add(cudnn.ReLU(true))                          -- ReLU activation function
    model:add(nn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
    model:add(cudnn.SpatialConvolution(32, 64, 3, 3))
    model:add(cudnn.SpatialMaxPooling(2,2,2,2))
    model:add(cudnn.ReLU(true))
    model:add(nn.SpatialBatchNormalization(64))
    model:add(cudnn.SpatialConvolution(64, 32, 3, 3))
    model:add(nn.View(32*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
    model:add(nn.Linear(32*4*4, 256))            -- fully connected layer (matrix multiplication between input and weights)
    model:add(cudnn.ReLU(true))
    model:add(nn.Dropout(0.5))                      --Dropout layer with p=0.5
    model:add(nn.Linear(256, outputSize))            -- 10 is the number of outputs of the network (in this case, 10 digits)

    ]]
    
    else
    print('Unknown model type')
    cmd:text()
    error()
end



model:add(nn.LogSoftMax())  -- f_i(x) = exp(x_i - shift) / sum_j exp(x_j - shift)



model:cuda() --ship to gpu
print(tostring(model))

local w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement()) --over-specified model


---- ### Classification criterion

criterion = nn.ClassNLLCriterion():cuda()

---    ### predefined constants

batchSize = opt.batch

optimState = {
    learningRate = 0.1
    
}

-- ### Main evaluation + training function

function dataAugmentation(data)
    return data
end

function forwardNet(data, labels, train)
    timer = torch.Timer()
    
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        x = dataAugmentation(x)
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.sgd(feval, w, optimState)
        end
    end

 

 
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    print(timer:time().real .. ' seconds')

    return avgLoss, avgError, tostring(confusion)
end


--- ### Train the network on training set, evaluate on separate set


epochs = opt.epoch

trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

if opt.load == 0 then
for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end
end




---        ### Introduce momentum, L2 regularization
--reset net weights

model:apply(function(l) l:reset() end)

optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = 1e-3
    
}
for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
end

print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs])
print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs])


-- save the model
if opt.save == 1 then
    print('Save the model:' .. filename)
    torch.save(filename, model		)
end


-- ********************* Plots *********************
require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('Loss.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('Error.png')
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()



end 




