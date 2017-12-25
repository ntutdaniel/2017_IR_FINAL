# coding: utf-8
from pip._vendor.distlib.compat import raw_input
import vector_space_model as vsm
import ir_file as ir_f
import copy

if __name__ == '__main__':
    pd = '../dataset/Document_final'
    po = '../dataset/Output'
    pq = '../dataset/TestQuery_final'
    d_start_index = 3
    q_start_index = 0
    e = 0.3
    qr_c = 200
    is_hw01 = True

    R_c = 2
    A = 0.5
    B = 0.5

    # map & hw13 -> 2314

    # 2422 -> 2421(10)
    # 2113 -> 2113(10) v 10 7 4
    # 2222 -> 2221(10)
    # 6661 -> 6661(10)
    # 5351 -> 5351(10)

    # tw-idf(tf-idf) 7621 => map@200 = 0.29728
    # tw-idf(bm25) 8621 => map@200 = 0.31552

    '''
    讀取檔案
    '''
    doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    d_c = len(doc_word_count)
    print('If run all recommended TF-IDF weighting schemes(y/n)')
    all_c = raw_input('(y/n): ')

    print('Input each query retrive rank max count')
    qr_c = int(raw_input('(0-' + str(d_c) + '): '))

    if (all_c == 'n'):
        # document tf
        print('1. Choose tf(i,j) method')
        print('(1.) {0, 1}')
        print('(2.) tf(i,j)')
        print('(3.) 1 + log2(tf(i,j))')
        print('(4.) (0.5 + 0.5 * (tf(i,j)/maxj(tf(i,j))))')
        print('(5.) (e + (1 - e) * (tf(i,j)/maxj(tf(i,j))))')
        print('(6.) sm25')
        print('(7.) tw-idf(tf-idf)')
        print('(8.) tw-idf(bm25)')
        d_tf_c = int(raw_input('d_tf_c(1-8): '))
        while (d_tf_c > 8 or d_tf_c < 1): d_tf_c = int(raw_input('try d_tf_c(1-8): '))

        # query tf
        print('2. Choose tf(i,q) method')
        print('(1.) {0, 1}')
        print('(2.) tf(i,q)')
        print('(3.) 1 + log2(tf(i,q))')
        print('(4.) (0.5 + 0.5 * (tf(i,q)/maxq(tf(i,q))))')
        print('(5.) (e + (1 - e) * (tf(i,q)/maxq(tf(i,q))))')
        print('(6.) sm25')
        q_tf_c = int(raw_input('q_tf_c(1-6): '))
        while (q_tf_c > 6 or q_tf_c < 1): q_tf_c = int(raw_input('try q_tf_c(1-6): '))

        if (d_tf_c == 5 or q_tf_c == 5):
            print('Input parameter e')
            e = float(raw_input("d_tf_c's e(0-1): "))
            while (e > 1 or e < 0): e = float(raw_input("e(0-1): "))

        # document idf
        print('3. Choose idf(i,j) method')
        print('(1.) 1')
        print('(2.) log(N/ni)')
        print('(3.) log(1 + N/ni)')
        print('(4.) log(1 + maxi(ni)/ni)')
        print('(5.) log((N - ni)/ni)')
        print('(6.) sm25')
        d_idf_c = int(raw_input('d_idf_c(1-5): '))
        while (d_idf_c > 6 or d_idf_c < 1): d_idf_c = int(raw_input('try d_idf_c(1-6): '))

        # document idf
        print('4. Choose idf(i,q) method')
        print('(1.) 1')
        print('(2.) log(N/ni)')
        print('(3.) log(1 + N/ni)')
        print('(4.) log(1 + maxi(ni)/ni)')
        print('(5.) log((N - ni)/ni)')
        q_idf_c = int(raw_input('q_idf_c(1-5): '))
        while (q_idf_c > 5 or q_idf_c < 1): q_idf_c = int(raw_input('try q_idf_c(1-5): '))

        print('processing...')

        q_w_c = copy.deepcopy(query_word_count)
        q_w_c1 = copy.deepcopy(query_word_count)
        # q_w_c2 = copy.deepcopy(query_word_count)

        d_w_c = copy.deepcopy(doc_word_count)
        d_w_c1 = copy.deepcopy(doc_word_count)
        d_w_c2 = copy.deepcopy(doc_word_count)

        d_w_c_d = copy.deepcopy(folder_word_count_distinct)
        # d_w_c_d1 = copy.deepcopy(folder_word_count_distinct)
        d_w_c_d2 = copy.deepcopy(folder_word_count_distinct)

        sim_q = vsm.calDocumantRank(d_w_c, d_w_c_d, q_w_c, po, d_tf_c, q_tf_c, d_idf_c,
                                    q_idf_c, e, qr_c)
        q_ro = vsm.Rocchio(sim_q, q_w_c1, d_w_c1, R_c, A, B)
        sim_q = vsm.calDocumantRank(d_w_c2, d_w_c_d2, q_ro, po, d_tf_c, q_tf_c, d_idf_c,
                                    q_idf_c, e, qr_c)
        vsm.outputfile(sim_q, is_hw01, po, d_tf_c, q_tf_c, d_idf_c,
                       q_idf_c, qr_c)
    else:
        for i in range(1, 7, 1):
            for j in range(1, 7, 1):
                for k in range(1, 7, 1):
                    for l in range(1, 6, 1):
                        print((i - 1) * 5 ** 3 + (j - 1) * 5 ** 2 + (k - 1) * 5 ** 1 + (l - 1))
                        q_w_c = copy.deepcopy(query_word_count)
                        q_w_c1 = copy.deepcopy(query_word_count)
                        # q_w_c2 = copy.deepcopy(query_word_count)

                        d_w_c = copy.deepcopy(doc_word_count)
                        d_w_c1 = copy.deepcopy(doc_word_count)
                        d_w_c2 = copy.deepcopy(doc_word_count)

                        d_w_c_d = copy.deepcopy(folder_word_count_distinct)
                        # d_w_c_d1 = copy.deepcopy(folder_word_count_distinct)
                        d_w_c_d2 = copy.deepcopy(folder_word_count_distinct)

                        sim_q = vsm.calDocumantRank(d_w_c, d_w_c_d, q_w_c, po, i, j, k, l, e, qr_c)
                        q_ro = vsm.Rocchio(sim_q, q_w_c1, d_w_c1, R_c, A, B)
                        sim_q = vsm.calDocumantRank(d_w_c2, d_w_c_d2, q_ro, po, i, j, k, l, e, qr_c)
                        vsm.outputfile(sim_q, is_hw01, po, i, j, k, l, qr_c)

    print('done!')
