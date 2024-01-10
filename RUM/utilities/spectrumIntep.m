function  spectrumIntep =  spectrumIntep (energy0,data)
%%
% Subspace Modeling for Fast and High-sensitivity X-ray Chemical Imaging

%%
energy = linspace(min(energy0),max(energy0),5000);
[M,P]=size(data);
parfor i=1:M
   spectrum=smooth(data(i,:));
   spectrum=spectrum-min(spectrum);
   spectrum=spectrum./spectrum(end) ;
   tmp=interp1([1:size(spectrum,1)],spectrum,linspace(1,size(spectrum,1),5000))'; 
   spectrumIntep(i,:)=smooth(tmp);
end
end        