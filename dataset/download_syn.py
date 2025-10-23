# Description: This script downloads the data from Synapse to a local folder.
import synapseclient
import synapseutils

def download_data( local_folder):
    print("Start downloading")

    # login to Synapse
    syn = synapseclient.Synapse()

    syn.login(authToken='eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc0MTc5NDk2MSwiaWF0IjoxNzQxNzk0OTYxLCJqdGkiOiIxNzQ1OSIsInN1YiI6IjM1MDI0NzQifQ.P6Q0O1amJLgFbgkcnQi7NESpNNIPf0BtvME4aKxbK--adyYbACb0Q3HrFz7J_K8bweDJw4189c6dXV9g_aHkRZLcNmYlnGHj_AvqWOYFJsAq2NcohDPnNQ3pK1zP1Ud8zYr092QAxZbYUzDOf0Kbcw8TQOFo8q5xYWrJ0_MhANtOpIUp-Q4vZ25pbRPtpBAO5MOR5815NwqpZYv4jpdHc29-fp_YFAMgb9dMZXj3gmYAt4_H7xKrp69nsF0-bvOnUm7c50UOlRo1nQ9phVYfdSdDnYGnPPkWsFmAQn4U-fMydgWv7SCgKyQxvugFcpyA4p894hQLr9-iGV69s-gx4Q')

    # syn = synapseclient.login(email=email, password=password, rememberMe=True)

    # download all the files in folder files_synapse_id to a local folder
    project_id = "syn21903917" # this is the project id of the files.
    all_files = synapseutils.syncFromSynapse(syn, entity=project_id, path=local_folder)

    print("Finished downloading")


if __name__ == "__main__":

    # settings
    local_folder = '/media/guiqiu/Weakly_supervised_data/HeiCO'
    email = "<email>"
    password = "<password>"

    # download data
    download_data(  local_folder)
# syn = synapseclient.Synapse()
# syn.login(authToken='eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc0MTc5NDk2MSwiaWF0IjoxNzQxNzk0OTYxLCJqdGkiOiIxNzQ1OSIsInN1YiI6IjM1MDI0NzQifQ.P6Q0O1amJLgFbgkcnQi7NESpNNIPf0BtvME4aKxbK--adyYbACb0Q3HrFz7J_K8bweDJw4189c6dXV9g_aHkRZLcNmYlnGHj_AvqWOYFJsAq2NcohDPnNQ3pK1zP1Ud8zYr092QAxZbYUzDOf0Kbcw8TQOFo8q5xYWrJ0_MhANtOpIUp-Q4vZ25pbRPtpBAO5MOR5815NwqpZYv4jpdHc29-fp_YFAMgb9dMZXj3gmYAt4_H7xKrp69nsF0-bvOnUm7c50UOlRo1nQ9phVYfdSdDnYGnPPkWsFmAQn4U-fMydgWv7SCgKyQxvugFcpyA4p894hQLr9-iGV69s-gx4Q')
# dataset = syn.get('syn21903917', downloadLocation='C:/2data/HeiCO')
