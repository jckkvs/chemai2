<template>
  <div class="data-upload-container">
    <h2>📂 データ読み込み</h2>
    
    <el-card class="upload-card">
      <el-upload
        drag
        :auto-upload="true"
        :on-success="handleSuccess"
        :on-error="handleError"
        :action="apiUrl"
        accept=".csv,.xlsx,.xls"
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          CSV / Excel をドラッグ&ドロップまたは<em>クリックしてアップロード</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            csv/xlsx ファイルのみ対応
          </div>
        </template>
      </el-upload>

      <div v-if="dataInfo" class="data-info">
        <el-divider>読み込んだデータ</el-divider>
        
        <el-descriptions :column="2" border>
          <el-descriptions-item label="ファイル名">
            {{ dataInfo.filename }}
          </el-descriptions-item>
          <el-descriptions-item label="行数">
            {{ dataInfo.rows?.toLocaleString() }}
          </el-descriptions-item>
          <el-descriptions-item label="列数">
            {{ dataInfo.cols }}
          </el-descriptions-item>
          <el-descriptions-item label="目的変数">
            <el-tag>{{ dataInfo.target_col }}</el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <el-divider>データプレビュー</el-divider>
        <el-table :data="dataInfo.preview" border stripe max-height="300">
          <el-table-column
            v-for="col in dataInfo.columns"
            :key="col"
            :prop="col"
            :label="col"
            :min-width="100"
          />
        </el-table>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

const apiUrl = 'http://localhost:8000/api/upload'
const dataInfo = ref<any>(null)

const handleSuccess = (response: any) => {
  ElMessage.success(response.message || 'アップロード成功')
  dataInfo.value = response
}

const handleError = (error: any) => {
  ElMessage.error(`アップロード失敗: ${error.message}`)
}
</script>

<style scoped>
.data-upload-container {
  max-width: 1200px;
  margin: 0 auto;
}

.upload-card {
  margin-top: 20px;
}

.el-icon--upload {
  font-size: 67px;
  color: #8c939d;
  margin: 40px 0 16px;
}

.data-info {
  margin-top: 20px;
}
</style>
