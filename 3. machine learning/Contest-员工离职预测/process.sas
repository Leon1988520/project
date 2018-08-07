/*
	1. Load Orig File
	
	数据导入mysql，再读入sas，不方便之处在于，mysql要逐个对变量的数据类型进行设置，变量多的情况下，是很难处理的
	而sas读入数据，有自动数据类型检验功能(猜的)
*/

PROC IMPORT OUT= WORK.orig_pfm_train
            DATAFILE= "E:\workStation\datamining\Contest-员工离职预测\pfm_train.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;

PROC IMPORT OUT= WORK.orig_pfm_test
            DATAFILE= "E:\workStation\datamining\Contest-员工离职预测\pfm_test.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;


/*
	2. 校正数据类型
*/

DATA pfm_train(
		RENAME = (
			age_num = age
	));
	FORMAT age_num 2.;
	SET orig_pfm_train;

	age_num = age;
	DROP age;
RUN;

DATA pfm_test(
		RENAME = (
			age_num = age
	));
	FORMAT age_num 2.;

	SET orig_pfm_test;
	age_num = age;
	DROP age;
RUN;

