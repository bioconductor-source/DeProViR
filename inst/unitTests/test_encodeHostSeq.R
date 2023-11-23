test_encodeHostSeq = function() {
   embeddings_index <- gloveImport()
   dt <- loadTrainingSet()
   encoded_seq <- encodeHostSeq(dt, embeddings_index)
   checkTrue(nrow(encoded_seq[["embedding_matrix_h"]]) == 20)
}
