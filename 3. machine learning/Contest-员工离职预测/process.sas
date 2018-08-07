/*
	1. Load Orig File
	
	���ݵ���mysql���ٶ���sas��������֮�����ڣ�mysqlҪ����Ա������������ͽ������ã������������£��Ǻ��Ѵ����
	��sas�������ݣ����Զ��������ͼ��鹦��(�µ�)
*/

PROC IMPORT OUT= WORK.orig_pfm_train
            DATAFILE= "E:\workStation\datamining\Contest-Ա����ְԤ��\pfm_train.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;

PROC IMPORT OUT= WORK.orig_pfm_test
            DATAFILE= "E:\workStation\datamining\Contest-Ա����ְԤ��\pfm_test.csv" 
            DBMS=CSV REPLACE;
     GETNAMES=YES;
     DATAROW=2; 
RUN;


/*
	2. У����������
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

