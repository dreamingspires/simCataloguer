from make_writer import Writer

writer = Writer()
writer.train_model('obama_sotu.txt', 'simObama')
print(writer.generate(run_name='simObama'))