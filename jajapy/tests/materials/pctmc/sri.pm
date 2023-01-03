ctmc
const double beta;
const double gamma;
const double plock = 0.472081;
const int SIZE = 1000;
const int BOUND = 200;

module parameters
	[new_recovery] true -> gamma : true;
	[new_infection] true -> beta : true;
endmodule

module SIR_Pisa
	i : [0..BOUND] init 48;
	r : [0..BOUND] init 16;
	[new_recovery]  i>0 & r<BOUND -> i*plock : (i'=i-1) & (r'=r+1);
	[new_recovery]  i>0 & r=BOUND -> i*plock : (i'=i-1);
	[new_infection] i>0 & i<BOUND -> (SIZE-(i+r))*i*plock/SIZE : (i'=i+1);
endmodule


label "i_0" = i <=20;
label "i_1" = i > 20  & i <= 40;
label "i_2" = i > 40  & i <= 60;
label "i_3" = i > 60  & i <= 80;
label "i_4" = i > 80  & i <= 100;
label "i_5" = i > 100 & i <= 120;
label "i_6" = i > 120 & i <= 140;
label "i_7" = i > 140 & i <= 160;
label "i_8" = i > 160 & i <= 180;
label "i_9" = i > 180 & i <= 200;


label "r_0" = r <=20;
label "r_1" = r > 20  & r <= 40;
label "r_2" = r > 40  & r <= 60;
label "r_3" = r > 60  & r <= 80;
label "r_4" = r > 80  & r <= 100;
label "r_5" = r > 100 & r <= 120;
label "r_6" = r > 120 & r <= 140;
label "r_7" = r > 140 & r <= 160;
label "r_8" = r > 160 & r <= 180;
label "r_9" = r > 180 & r <= 200;
