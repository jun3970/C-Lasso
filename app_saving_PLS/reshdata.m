% reshape data

Dret_rf  = stkest(:,1);
mkt_rf   = stkest(:,2);
SMB = stkest(:,3);
VMG = stkest(:,4);

% FivSMB = stkest(:,5);
% FivHML = stkest(:,6);
% FivRMW = stkest(:,7);
% FivCMA = stkest(:,8);

Index = stkestMATLABindex;
code = Index(:,1);
date  = Index(:,2);

%%
clear('stkest', 'stkestMATLABindex', 'Index')

save CH3-2017-09-30.mat



