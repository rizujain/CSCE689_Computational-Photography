clear; clc; close all;

%%% Assignment 2 - Starter code

% Setting up the input output paths
inputDir = '../Images/';
outputDir = '../Results/';

% False for source gradient, true for mixing gradients
isMix = false;

N = 8;
offsets = cell(N,1);
offsets{1} = [0, 0];
offsets{2} = [210, 10];
offsets{3} = [10, 28];
offsets{4} = [140, 80];
offsets{5} = [-40, 90];
offsets{6} = [60, 100];
offsets{7} = [20, 20];
offsets{8} = [-28, 88];

for i = 1 : N
    
    source = im2double(imread(sprintf('%s/source_%02d.jpg', inputDir, i)));
    mask = im2double(imread(sprintf('%s/mask_%02d.jpg', inputDir, i)));
    target = im2double(imread(sprintf('%s/target_%02d.jpg', inputDir, i)));
    
    % Align the source and mask using the provided offest
    [source, mask] = AlignImages(source, mask, target, offsets{i});
    
    % cleaning up the mask
    mask(mask < 0.5) = 0;
    mask(mask >= 0.5) = 1;
    
    
    %%%% The main part of the code. 
    
    % Implement the PyramidBlend function (Task 1)
    pyramidOutput = PyramidBlend(source, mask, target);
    
    % Implement the PoissonBlend function (Task 2)
    poissonOutput = PoissonBlend(source, mask, target, isMix);

    
    % Writing the result
    
    pyramidOutput = im2uint8(pyramidOutput);
    imwrite(pyramidOutput, sprintf('%s/pyramid_%02d.jpg', outputDir, i));
    
    poissonOutput = im2uint8(poissonOutput);
    if (~isMix)
        imwrite(poissonOutput, sprintf('%s/poisson_%02d.jpg', outputDir, i));
    else
        imwrite(poissonOutput, sprintf('%s/poisson_%02d_Mixing.jpg', outputDir, i));
    end
    
end

