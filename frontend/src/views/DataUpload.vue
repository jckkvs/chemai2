<template>
  <div class="data-upload">
    <el-card>
      <template #header>
        <div class="card-header">
          <el-icon><Upload /></el-icon>
          <span>データ読み込み</span>
        </div>
      </template>

      <!-- アップロードエリア -->
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

      <!-- データ情報表示 -->
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
            {{ dataInfo.columns_count }}
          </el-descriptions-item>
          <el-descriptions-item label="目的変数">
            <el-tag>{{ dataInfo.target_col }}</el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <!-- 列設定 -->
        <el-divider>列の役割設定</el-divider>
        <el-form :model="columnConfig" label-width="120px">
          <el-form-item label="目的変数">
            <el-select v-model="columnConfig.target_col" placeholder="選択">
              <el-option
                v-for="col in dataInfo.columns"
                :key="col.name"
                :label="col.name"
                :value="col.name"
              />
            </el-select>
          </el-form-item>
          
          <el-form-item label="タスクタイプ">
            <el-radio-group v-model="columnConfig.task_type">
              <el-radio label="regression">回帰</el-radio>
              <el-radio label="classification">分類</el-radio>
            </el-radio-group>
          </el-form-item>
          
          <el-form-item>
            <el-button type="primary" @click="updateColumns">
              設定を適用
            </el-button>
          </el-form-item>
        </el-form>

        <!-- データプレビュー -->
        <el-divider>データプレビュー</el-divider>
        <el-table :data="dataInfo.sample_data" border stripe max-height="300">
          <el-table-column
            v-for="col in dataInfo.columns"
            :key="col.name"
            :prop="col.name"
            :label="col.name"
            :min-width="120"
          />
        </el-table>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled, Upload } from '@element-plus/icons-vue'
import axios from 'axios'

const apiUrl = 'http://localhost:8000/api/data/upload'
const dataInfo = ref<any>(null)
const columnConfig = ref({
  target_col: '',
  task_type: 'regression'
})

const handleSuccess = (response: any) => {
  ElMessage.success(response.message)
  fetchDataInfo()
}

const handleError = (error: any) => {
  ElMessage.error(`アップロード失敗: ${error.message}`)
}

const fetchDataInfo = async () => {
  try {
    const response = await axios.get('http://localhost:8000/api/data/info')
    dataInfo.value = response.data
    columnConfig.value.target_col = response.data.target_col
    columnConfig.value.task_type = response.data.task_type
  } catch (error) {
    // データ未読み込み時は何もしない
  }
}

const updateColumns = async () => {
  try {
    await axios.post('http://localhost:8000/api/data/columns', columnConfig.value)
    ElMessage.success('列設定を更新しました')
    fetchDataInfo()
  } catch (error) {
    ElMessage.error('設定更新に失敗しました')
  }
}

onMounted(() => {
  fetchDataInfo()
})
</script>

<style scoped>
.data-upload {
  max-width: 1200px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 18px;
  font-weight: bold;
}

.data-info {
  margin-top: 20px;
}

.el-icon--upload {
  font-size: 67px;
  color: #8c939d;
  margin: 40px 0 16px;
}
</style>
