Examples of the code used to get the citation numbers. The scrape_gScholar_3_3.py script in each sub folder loads the dataframe containing papers for which to get the citations (dfSplitInput0.csv). Entries which have already been scrapped are stored as pickled dictionaries in the "prevDone" folder. The script removes papers for which it has previously gotten the citation number, then starts scrapping. 

only 2 of ~25 blocks are actually shown here

The setUpParRuns.py script contains functions to create all the subfolders, divide up the papers to be scrapped, and aggregate the results from each subfolder to a single dataframe after the scrapping is done.
