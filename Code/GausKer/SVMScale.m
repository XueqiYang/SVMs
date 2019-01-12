function [ dataset_scale ] = SVMScale( dataset ,min,max)
%SVMScale scales dataset to [min,max]
%By default, min = 0, max =1
    if(nargin <2)
        min = 0;
        max = 1;
    end
    
    [mset,nset] = size(dataset);
    [dataset_scale,ps] = mapminmax(dataset',min,max);
    dataset_scale = dataset_scale';
end

