function imgt = aux_imscale(img, range)
%%
% Subspace Modeling for Fast and High-sensitivity X-ray Chemical Imaging
%%
% Check params:
if nargin<2
    range=[0,1];
end

% Transform the image type:
imgt = double(img);

% Obtain and appling the offset:
imgt = imgt-min(min(min(imgt)));

% Obtain and appling the scale:
scale = max(max(max(imgt)));
if scale>0
    imgt = imgt/scale;
end

% Generate the required image:
if not(abs(range(2)-range(1))==1)
    imgt = imgt*abs(range(2)-range(1));
end
if not(range(1)==0)
    imgt = imgt+range(1);
end