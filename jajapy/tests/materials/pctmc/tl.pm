ctmc

const double p;
const double q;

module car_tl
	cs : [0..2] init 0;
	[nothing] cs=0  -> 0.5 : (cs'=2);
	[button] cs=2  -> q : (cs'=1);
	[] cs=1 -> q : (cs'=0);
endmodule

module ped_tl
	ps : [0..1] init 0;
	[nothing] ps=0  -> 10.0 : (ps'=0);
	[button] ps=0  -> p/2 : (ps'=1);
	[] ps=1 -> q : (ps'=0);
endmodule

label "c_red"    = cs=0;
label "c_orange" = cs=1;
label "c_green"  = cs=2;
label "p_red"    = ps=0;
label "p_green"  = ps=1;