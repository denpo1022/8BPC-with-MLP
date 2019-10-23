def parallel_swar(i):  # from http://p-nand-q.com/python/algorithms/math/bit-parity.html
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    return int(i % 2)


inputs = [f'{j:08b}' for j in range(256)]
answers = [parallel_swar(j) for j in range(256)]
fp = open('inputs.txt', 'w')
fp.write('%s' % str(inputs))
fp.close()
fp = open('answer.txt', 'w')
fp.write('%s' % str(answers))
fp.close()
