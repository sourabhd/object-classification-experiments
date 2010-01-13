
%%
%% @file
%% @author Sourabh Daptardar <saurabh.daptardar@gmail.com>
%% @version 1.0
%%
%% @section LICENSE
%% This file is part of SerrePoggioClassifier.
%% 
%% SerrePoggioClassifier is free software: you can redistribute it and/or modify
%% it under the terms of the GNU General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% (at your option) any later version.
%% 
%% SerrePoggioClassifier is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with SerrePoggioClassifier.  If not, see <http://www.gnu.org/licenses/>.
%%
%% @section DESCRIPTION
%% This code has been developed in partial fulfillment of the M.Tech thesis
%% "Explorations on a neurologically plausible model of image object classification"
%% by Sourabh Daptardar, Y7111009, CSE, IIT Kanpur.
%%
%% This code implements normalized graph cut clustering technique.
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function display(inpfile,outputdir,title,clusters,strength,heapordering)

	[clusters,strength,heapordering] = sortclusters(clusters,strength,heapordering);

	clusters
	strength
	heapordering

	OSSEP = "/";
	%% Read keyfile
	filelist = {};
	ifd = fopen(inpfile,'r');
	while ( (line = fgets(ifd)) != -1 )
		[key,file] = sscanf(line,'I%d=%s','C');
		filelist(key+1) = file;
	endwhile
	fclose(ifd);

	
	indexfile = sprintf('%s%sindex.htm',outputdir,OSSEP); 
	imagedirel = "images";
	imagedir = sprintf('%s%s%s',outputdir,OSSEP,imagedirel); 
	[status, msg, msgid] = mkdir(outputdir);
	[status, msg, msgid] = mkdir(imagedir);
	ofd = fopen(indexfile,'w'); 
	if ( ofd == -1 )
		fprintf(stderr,'Could not open file %s',indexfile);
		exit(1);
	endif

	fprintf(ofd,'<html>\n\t<head>\n\t\t<title>%s</title>\n\t</head>\n\t<body>\n\t\t<h2>%s</h2>\n\t\t<table border="1">',title,title);
	fprintf(ofd,'\n\t\t\t<tr>\n\t\t\t\t<th>%s</th>\n\t\t\t\t<th>%s</th>\n\t\t\t\t<th>%s</th>\n\t\t\t\t<th><i><font size="1">%s</font></i></th>\n\t\t\t\t</tr>','Cluster Num','Cluster Strength','Number of Images','(Click on the images below)' );

	sz = size(clusters,2)
	for i = 1:sz
		fname = sprintf('%s%s%d.htm',outputdir,OSSEP,i); 
		fnamerel = sprintf('%d.htm',i); 
		fdi =  fopen(fname,'w');
		if ( fdi == -1 )
			fprintf(stderr,'Could not open file %s',fname);
			exit(1);
		endif
		
				fprintf(fdi,'<html>\n\t<head>\n\t\t<title>%s</title>\n\t</head>\n\t<body>\n\t\t<h2>Cluster %d (ID: %d)</h2>\n\t\t<font size="1"><i>(Click images to enlarge)</i></font>\n\t\t<table border="0">',title,i,cell2mat(heapordering(1,i)));

		p = cell2mat(clusters(1,i));
		printf('Display : Partition %d :\n',i);
		sz2 = size(p,2);
		rp = randperm(sz2);
		for j = 1:sz2
			imgfile = cell2mat(filelist(p(j)));
			[dir, name, ext, ver] = fileparts(imgfile);
			tgtfile = sprintf('%s%s%s%s',imagedir,OSSEP,name,ext);
			tgtfilerel = sprintf('%s%s%s%s',imagedirel,OSSEP,name,ext);
			[status, msg, msgid] = copyfile(imgfile, tgtfile, 'f');

			fprintf(fdi,'\n\t\t\t<tr><td><a href="%s"><img src="%s" alt="%s" height="100" width="100" /></a></td></tr>',tgtfilerel,tgtfilerel,tgtfilerel);
%			if ( j == 1 )
			if ( j == rp(1) )
				fprintf(ofd,'\n\t\t\t<tr>\n\t\t\t\t<td>%d (ID: %d)</td>\n\t\t\t\t<td>%f</td>\n\t\t\t\t<td>%d</td>\n\t\t\t\t<td><center><a href="%s"><img src="%s" alt="%s" height="100" width="100" /><center></a></td>\n\t\t\t</tr>',i,cell2mat(heapordering(1,i)),strength(i),sz2,fnamerel,tgtfilerel,tgtfilerel);
			endif
		endfor

		printf('Display : Partition %d strength : %f\n\n',i,strength(i));

		fprintf(fdi,'\n\t\t</table>\n\t</body>\n</html>');

		fclose(fdi);
	
	endfor 
	
	fprintf(ofd,'\n\t\t</table>\n\t</body>\n</html>');
	
	fclose(ofd);

endfunction

function [cl,st,ho] = sortclusters(clusters,strength,heapordering)
	cls = clusters; % make a copies
	str = strength;
	hor = heapordering;
	hor
	cl = {};
	st = [];
	ho = {};
	sz = size(cls,2);
	counter = 0;
	while( sz > 0 )
		maxsz = 0;
		maxindex = 0;
		for i = 1:sz
			p = cell2mat(cls(1,i));
			sz2 = size(p,2);
			if( sz2 > maxsz )
				maxsz = sz2;
				maxindex = i;
			endif
		endfor;
		counter = counter+1;
		cl(1,counter) = cls(:,maxindex); % reordering 
		st(1,counter) = str(maxindex);
		ho(1,counter) = hor(:,maxindex);
		cls(maxindex) = '';    % deletion 
		str(maxindex) = '';
		hor(maxindex) = '';
		sz = size(cls,2);
	endwhile
endfunction



