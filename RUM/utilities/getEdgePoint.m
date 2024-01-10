function edgepointmap = getEdgePoint(energy0,data)
%%
% Subspace Modeling for Fast and High-sensitivity X-ray Chemical Imaging

%%
energy = linspace(min(energy0),max(energy0),5000);
[M,N,P]=size(data);
data1=reshape(data,[M*N,P]);
parfor i=1:M*N
   spectrum=smooth(data1(i,:));
   spectrum=spectrum-min(spectrum);
   spectrum=spectrum./spectrum(end);
   
   spectrumIntep=interp1([1:size(spectrum,1)],spectrum,linspace(1,size(spectrum,1),5000))';
   
   [c index] = min(abs(spectrumIntep-0.5));
   
   
   edgepointmap(i) = energy(index);
end
edgepointmap=reshape(edgepointmap,[M,N]);

end