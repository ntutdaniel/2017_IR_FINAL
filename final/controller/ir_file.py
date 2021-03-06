# coding: utf-8
import os
import time
# calculate the word count in document
def ReadFile(path, start_index):
    word_counter = {}
    word_tw_grah = {}       #建圖 [word3]={word1,word2}
    word_tw = {}            #從上面的圖取得in_degree數量
    TW_WINDOW_SIZE = 2      #不包含自己 看前幾個
    document_info = []
    document_word_count = 0

    for i, line in enumerate(open(path, 'r')):
        word_list = line.replace('\n', '').replace('-1', '').strip()

        if (i < start_index):
            document_info.append(word_list)
        else:
            #for word in word_list.split():
            for index,word in enumerate(word_list.split()):
                #print(word_list.split())
                temp = str(word)
                document_word_count += 1
                if temp not in word_counter:
                    word_counter[temp] = 1
                else:
                    word_counter[temp] += 1
                    
                    #--- add_for_tw
                if(index>=TW_WINDOW_SIZE):
                    for k in range(index-TW_WINDOW_SIZE,index):
                        #print(index,word,word_list.split()[k])                        
                        temp_addtw=word_list.split()[k]
                        if temp in word_tw_grah :                            
                            word_tw_grah[temp].add(temp_addtw)
                        else:
                            word_tw_grah[temp]={temp_addtw}
                            
                    word_tw[temp]=len(word_tw_grah[temp])
                
                
                    
    return document_info, word_counter, document_word_count, word_tw


# debug
def ReadFileDebug(document_info, word_counter):
    for info in document_info:
        print(info)
    print('-' * 18)
    print('{:15}{:3}'.format('Word', 'Count'))
    print('-' * 18)
    for (word, occurance) in word_counter.items():
        print('{:15}{:3}'.format(word, occurance))


def ReadFolder(p, start_index):
    folder = {}
    folder_tw={}
    files = os.listdir(p)
    files_name = []
    folder_word_counter = {}
    folder_word_counter_distince = {}  # word出現在各個document的次數（如果出現為1）

    for f in files:  # f is file name
        fullpath = os.path.join(p, f)
        if os.path.isfile(fullpath):
            files_name.append(f)

    # read all files in the folder
    for fn in files_name:
        fullpath = os.path.join(p, fn)
        if (fn == '.DS_Store'): continue
        document_info, word_counter, document_word_count, word_tw = ReadFile(fullpath, start_index)

        folder[fn] = word_counter
        folder_tw[fn] = word_tw
        # debug
        # if debug: ReadFileDebug(document_info, word_counter)

    # files_count = len(folder)

    # calculate all word count in the folder
    for (f, words) in folder.items():
        for (word, count) in words.items():
            if word not in folder_word_counter:
                folder_word_counter[word] = count
                folder_word_counter_distince[word] = 1
            else:
                folder_word_counter[word] += count
                folder_word_counter_distince[word] += 1

    # doc word count/ folder world count/ folder world distinct count
    return folder, folder_word_counter, folder_word_counter_distince,folder_tw


def ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct):
    print('{:30}{:15}{:3}'.format('File Name', 'Word', 'Count'))
    print('-' * 20)
    for (fn, words) in doc_word_count.items():
        if (os.path.exists(os.path.join(po, fn))):
            os.remove(os.path.join(po, fn))
        f = open(os.path.join(po, fn), 'w')
        for (word, count) in words.items():
            f.writelines(str(word) + ": " + str(count) + '\n')
            print('{:30}{:15}{:3}'.format(fn, word, count))
        f.close()


def ReadEvaFile(path, start_index):
    ds = []

    for i, line in enumerate(open(path, 'r')):
        word_list = line.replace('\n', '').replace('-1', '').strip()
        ws = word_list.split(':')
        ds.append(ws[0])

    return ds


def ReadEvaFolder(p, start_index):
    folder = {}
    files = os.listdir(p)
    files_name = []
    folder_word_counter = {}
    folder_word_counter_distince = {}  # word出現在各個document的次數（如果出現為1）

    for f in files:  # f is file name
        fullpath = os.path.join(p, f)
        if os.path.isfile(fullpath):
            files_name.append(f)

    # read all files in the folder
    for fn in files_name:
        fullpath = os.path.join(p, fn)
        if (fn == '.DS_Store'): continue
        ds = ReadEvaFile(fullpath, start_index)

        folder[fn] = ds
        # debug
        # if debug: ReadFileDebug(document_info, word_counter)
    return folder


if __name__ == '__main__':
    debug1 = False
    debug2 = False
    debug3 = False

    p = '../dataset/Query/20001.query'
    pd = '../dataset/Document_final'
    po = '../dataset/Output'
    pq = '../dataset/TestQuery_final'
    
    start_index = 3

    # file
    #document_info, word_counter, document_word_count = ReadFile(p, start_index)
    # debug function
    if debug1: ReadFileDebug(document_info, word_counter)

    # documents
    doc_word_count, folder_word_count, folder_word_count_distinct,doc_folder_tw = ReadFolder(pd, start_index)
    # debug function
    if debug2: ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct,doc_folder_tw)

    # query
    start_index = 0
    doc_word_count, folder_word_count, folder_word_count_distinct,doc_folder_tw = ReadFolder(pq, start_index)
    # debug function
    if debug3: ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct,doc_folder_tw)
