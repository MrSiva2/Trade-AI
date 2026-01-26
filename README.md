# ISSUES TO BE ADDRESSED:
# 1. Preserving Logs
# 2. Fixing the bugs on the chart rendering on the backtesting page
# 3. Adding cache feature for all pages, so that the page loads faster when revisited after visiting other pages.
# 4. Any background processes (ie. training, backtesting) should not be paused or interrupted when opening other pages. Once the background process is completed, it should show a message pop up to notify the user, and the data should be cached.
# 5. The backtesteed data should be saved after the user clicks a save data button, and the data should be reflected on the dashboard.
# 6. Fixing the issue of model's name, where it is different form the name it is assigned when in training page.




# Here are your Instructions

# the updates that has cursor_v... in it is done by cursor. If something is not working,
# rollback to its previous version which does not have cursor in it

# the updates that has emergent_...V... in it is done from a diff acc.
# It is mentioned to just keep track to roll back and continue with different features.

# the "add file" commit has added the functionality of adding a csv file/ a new folder to the data management page.

# the "train_page" commit had added the training functionality to the train page. But, it is not saving the model's name as intended

# the "model_py" commit has divided the models into .py files and made them downloadable so that they can be edited or customised.

# the "bck_tst_1" commit has added backtesting functionality to the backtesting page. But, it is not saving the model name while training, and the backtesting page is not loading the charts to show the executions.

# the "bck_tst_2" commit has fixed the issue of the backtesting page not loading the charts to show the executions. But, it is laggy and showing the pre determined values to test.

# the "bck_tst_3" commit has fixed the issue of rendering the wrong price on chart.

