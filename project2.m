% Devin Yang %
% UCR ID: 862021705 %
% Project 2 - Feature Selection with Nearest Neightbor %

disp("Welcome to Devin Yang's Feature Selection Algorithm.");
inputFile = input('Type in the name of the file to test: ', 's');
dataset = load(inputFile); % Loads the data from the file
[numInstances, numFeatures] = size(dataset); % Returns the size of the matrix returned
numFeatures = numFeatures - 1; % Reduce by 1 feature to account for class column
disp(['This dataset has ', num2str(numFeatures), ' features (not including the class attribute) and ', num2str(numInstances), ' instances.']);
disp("Type the number of the algorithm you want to run: ");
disp("1) Forward Selection");
disp("2) Backward Elimination");
algChoice = input('Input your choice: ');
if algChoice == 1 % Forward Selection
    tic
    [bestAccuraccy, bestFeatures] = forwardSelect(dataset, numFeatures, numInstances);
    disp("Finished search.");
    disp(['The best feature subset is {', num2str(bestFeatures), '}, which has an accuracy of ', num2str(bestAccuraccy), '%.']);
    toc
elseif algChoice == 2 % Backward Elimination
    tic
    [bestAccuraccy, bestFeatures] = backwardElimination(dataset, numFeatures, numInstances);
    disp("Finished search.");
    disp(['The best feature subset is {', num2str(bestFeatures), '}, which has an accuracy of ', num2str(bestAccuraccy), '%.']);
    toc
else % Error input
    disp("Incorrect input, exiting.");
end

% Performs Forward Selection to determine the most accurate features in the dataset
function [bestAccuracy, bestFeatures] = forwardSelect(dataset, numFeatures, numInstances)
    disp("Beginning search.");
    fprintf('\n');
    currFeatures = [];
    bestFeatures = [];
    currAccuracy = 0;
    bestAccuracy = 0;
    tempAccuracy = 0;
    currAccuracy = LOOCrossValidation(dataset, currFeatures, numInstances) * 100; % Checks the accuracy of the empty list
    disp(['Feature set {', num2str(currFeatures) , '} has an accuracy of ', num2str(currAccuracy), '%.']);
    for i = 1:numFeatures
        disp(['On level ', num2str(i), ' of the search tree.']);
        disp(['Adding features to {', num2str(currFeatures), '}.'])
        bestsoFarAccuracy = 0;
        bestsoFarFeatures = [];
        for k = 1:numFeatures
            if isempty(intersect(currFeatures, k))
                currFeatures = [currFeatures, k]; % Appends a feature to the current feature set
                currAccuracy = LOOCrossValidation(dataset, currFeatures, numInstances) * 100;
                disp(['Using feature(s) {', num2str(currFeatures), '} with an accuracy of ', num2str(currAccuracy), '%.']);
                if currAccuracy > bestsoFarAccuracy % Updates the best so far accuracy and feature set on the current level of the search
                    bestsoFarAccuracy = currAccuracy;
                    bestsoFarFeatures = currFeatures;
                end
                if currAccuracy > bestAccuracy % Updates the best overall accuracy and feature set
                    bestAccuracy = currAccuracy;
                    bestFeatures = currFeatures;
                end
                currFeatures(end) = []; % Removes the added feature
            end
        end
        currFeatures = bestsoFarFeatures; % Sets the current feature set to the set that had the highest accuracy on that level
        disp(['Feature set {', num2str(currFeatures) , '} was best, with an accuracy of ', num2str(bestsoFarAccuracy), '%.']);
        if (bestsoFarAccuracy < tempAccuracy) % Notes if the accuracy decreased from the previous level
           disp("Warning: Accuracy has decreased from previous best.");
        end
        tempAccuracy = bestsoFarAccuracy;
        fprintf('\n');
    end
end

% Performs Backward Elimination to determine the most accurate features in the dataset
function [bestAccuracy, bestFeatures] = backwardElimination(dataset, numFeatures, numInstances)
    disp("Beginning search.");
    fprintf('\n');
    for i = 1:numFeatures % Initializes a 1xm array filled with all possible features
        currFeatures(i) = i;
    end
    bestFeatures = [];
    bestAccuracy = 0;
    tempAccuracy = 0;
    currAccuracy = LOOCrossValidation(dataset, currFeatures, numInstances) * 100; % Checks the accuracy of the whole list
    disp(['Feature set {', num2str(currFeatures) , '} has an accuracy of ', num2str(currAccuracy), '%.']);
    bestsoFarFeatures = bestFeatures;
    bestsoFarAccuracy = bestAccuracy;
    level = 1; 
    while ~isempty(currFeatures)
        disp(['On level ', num2str(level), ' of the search tree.']);
        disp(['Removing features from {', num2str(currFeatures), '}.'])
        level = level + 1; 
        if size(currFeatures, 2) > 1
            bestsoFarAccuracy = 0;
            bestsoFarFeatures = [];
            for i = 1:size(currFeatures,2)
                eliminated = currFeatures(i); % Keeps track of the removed feature
                currFeatures(i) = []; % Removes a feature from the current feature set
                currAccuracy = LOOCrossValidation(dataset, currFeatures, numInstances) * 100;
                disp(['Using feature(s) {', num2str(currFeatures), '} with an accuracy of ', num2str(currAccuracy), '%.']);
                if currAccuracy > bestsoFarAccuracy % Updates the best so far accuracy and feature set on the current level of the search
                    bestsoFarAccuracy = currAccuracy;
                    bestsoFarFeatures = currFeatures;
                end
                if currAccuracy > bestAccuracy % Updates the best overall accuracy and feature set
                    bestAccuracy = currAccuracy;
                    bestFeatures = currFeatures;
                end
                currFeatures = [currFeatures(1:i-1), eliminated, currFeatures(i:end)]; % Returns the removed feature to the feature set
            end
            currFeatures = bestsoFarFeatures; % Sets the current feature set to the set that had the highest accuracy on that level
        else % If there is only one more feature left to be eliminated
            currFeatures = [];
            bestsoFarAccuracy = LOOCrossValidation(dataset, currFeatures, numInstances) * 100;
            if bestsoFarAccuracy > bestAccuracy % Updates the best overall accuracy and feature set
                    bestAccuracy = bestsoFarAccuracy;
                    bestFeatures = currFeatures;
            end
        end
        disp(['Feature set {', num2str(currFeatures) , '} was best, with an accuracy of ', num2str(bestsoFarAccuracy), '%.']);
        if (bestsoFarAccuracy < tempAccuracy) % Notes if the accuracy decreased from the previous level
           disp("Warning: Accuracy has decreased from previous best.");
        end
        tempAccuracy = bestsoFarAccuracy; 
        fprintf('\n');
    end
end

% Calculates accuracy by Leave One Out Cross Validation
function accuracy = LOOCrossValidation(dataset, currFeatures, numInstances)
    correct = 0; % Number correctly classified
    for i = 1:numInstances
        nearestDistance = inf;
        nearestLocation = inf;
        for k = 1:numInstances
            if k ~= i 
                distance = 0; 
                for j = 1:numel(currFeatures) % Calculating Euclidean Distance
                    curr = currFeatures(j);
                    distance = distance + (dataset(k, curr + 1) - dataset(i, curr + 1))^2;
                end
                distance = sqrt(distance);
                if distance < nearestDistance
                    nearestDistance = distance;
                    nearestLocation = dataset(k,1);
                end
            end
        end
        if dataset(i,1) == nearestLocation
            correct = correct + 1;
        end
    end
    accuracy = correct/numInstances;
end
