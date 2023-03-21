file = open("E:\\sfx_1_33_targeting\\linearitycomp\\batch2\\Para_3-0175ft_PHI_41_THETA_63\\Para_3-0175ft_PHI_41_THETA_63_Site0_Orientation_4.ikic" , "r")
l=file.readlines() # splitst the ikic file into lines
file.close()
print(len(l)) #tells the number of lines in the ikic file.
for _ in range(len(l)):
    if l[_][0] == 'B':
        print('sup',_)

# have to do some stuff with pattern or [] to get it to find the correct entry. maybe lists? idk

 #InitiationToughnessFile.write(str(keq) + ',' + str(K1c_MTS) + ',' + str(K1c_MSS)
 #                                   + ',' + str(K1c_Final) + ',' + str(K_MTS) + ',' + str(K_MSS) +',' + str(K_Dominant) + ',' + str(K_diff) + '\n')  #### see if this works-bjorn